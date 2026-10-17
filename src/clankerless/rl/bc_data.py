"""Generate a supervised dataset of expert (F) decisions for behavior cloning.

Reinforcement learning from scratch has to discover Catan through a single win/loss bit
spread over ~84 moves. Behavior cloning instead learns from fully-labelled examples --
"in this position, F played this" -- which is far more sample-efficient, and gives the
policy a strong starting point that RL can then fine-tune (the AlphaGo recipe).

Only Color.BLUE's decisions are recorded: MOVE_ROBBER actions encode absolute victim
colors, so moves logged from RED's seat would decode to "rob myself" for a BLUE-seated
policy.

Run:  python -m src.clankerless.rl.bc_data --games 1000
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from catanatron import GameAccumulator, Color, Game
from catanatron.features import create_sample_vector, get_feature_ordering
from catanatron.gym.envs.action_space import get_action_array, to_action_space
from catanatron.models.player import RandomPlayer
from catanatron.players.value import ValueFunctionPlayer

from src.clankerless.agents.heuristic import HeuristicPlayer


class ExpertRecorder(GameAccumulator):
    """
    this class will record (observation, F's Action, Legality Mask) for one color's decisions
    """

    def __init__(self, target_color, features, player_colors, map_type='BASE'):
        self.masks = None
        self.acts = None
        self.obs = None
        self.target_color = target_color
        self.features = features
        self.player_colors = player_colors
        self.map_type = map_type
        self.n_actions = len(get_action_array(player_colors, map_type))

    def before(self, game):
        self.obs, self.acts, self.masks = [], [], []

    def step(self, game_before_action, action):
        if action.color != self.target_color:
            return

        playable = game_before_action.playable_actions

        if len(playable) <= 1:
            return

        self.obs.append(create_sample_vector(game_before_action, self.target_color, self.features))
        self.acts.append(to_action_space(action, self.player_colors, self.map_type))
        m = np.zeros(self.n_actions, dtype=bool)

        for a in playable:
            m[to_action_space(a, self.player_colors, self.map_type)] = True
        self.masks.append(m)


def _make_opponent(seed):
    """Pick the RED opponent for one game, mixing strengths to diversify states.

    Recording only F-vs-F games teaches the network F's moves in positions two F players
    reach; it would never see the messier boards its own play produces. Mixing in weaker
    opponents broadens the state distribution while the label is still always F's move.

    :param seed: the game seed, used to pick deterministically
    :return: a Catanatron Player for Color.RED
    """
    pick = seed % 20
    if pick < 12:
        return ValueFunctionPlayer(Color.RED)
    if pick < 17:
        return HeuristicPlayer(Color.RED)
    return RandomPlayer(Color.RED)


def _generate_one(seed):
    """Play one game and return F's recorded (observations, actions, masks) as BLUE.

    Runs in a worker process, so it takes only a picklable int and returns arrays.

    :param seed: game seed
    :return: tuple (obs [n,614] float32, actions [n] int64, masks [n,332] bool)
    """
    features = get_feature_ordering(2, "BASE")
    players = [ValueFunctionPlayer(Color.BLUE), _make_opponent(seed)]
    if seed % 2:
        players.reverse()   # alternate who moves first; BLUE stays BLUE
    rec = ExpertRecorder(Color.BLUE, features, (Color.BLUE, Color.RED))
    Game(players, seed=seed).play(accumulators=[rec])
    return (np.array(rec.obs, np.float32),
            np.array(rec.acts, np.int64),
            np.array(rec.masks, dtype=bool))


def generate(n_games=1000, workers=None, seed=0, out=None):
    """Play games in parallel and save F's decisions as a supervised dataset.

    :param n_games: number of games to play
    :param workers: worker processes (default: all CPUs)
    :param seed: base seed
    :param out: output .npz path (default: bc_data.npz beside this module)

    :return: the path written
    """
    out = out or os.path.join(os.path.dirname(__file__), "bc_data.npz")
    tasks = [seed + i for i in range(n_games)]
    n_workers = workers or mp.cpu_count()
    chunk = max(1, n_games // (n_workers * 4))

    print(f"generating {n_games} games on {n_workers} workers ...")
    t0 = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        results = pool.map(_generate_one, tasks, chunksize=chunk)

    results = [r for r in results if len(r[1]) > 0]
    obs = np.concatenate([r[0] for r in results])
    acts = np.concatenate([r[1] for r in results])
    masks = np.concatenate([r[2] for r in results])

    # F's chosen action must always have been legal; a failure here means the action
    # index mapping is wrong, which would poison every downstream label.
    assert masks[np.arange(len(acts)), acts].all(), "recorded action not marked legal in its mask"

    np.savez_compressed(out, obs=obs, actions=acts, masks=masks)
    dt = time.perf_counter() - t0
    print(f"{len(acts):,} samples from {len(results)} games in {dt:.1f}s "
          f"({len(acts) / max(1, len(results)):.1f} per game)")
    print(f"obs {obs.shape} {obs.dtype} | actions {acts.shape} | masks {masks.shape}")
    print(f"distinct actions used: {len(np.unique(acts))} of {masks.shape[1]}")
    print("saved ->", out)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate behavior-cloning data from F.")
    parser.add_argument("--games", type=int, default=1000, help="games to play")
    parser.add_argument("--workers", type=int, default=None, help="worker processes")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    parser.add_argument("--out", default=None, help="output .npz path")
    args = parser.parse_args()
    generate(n_games=args.games, workers=args.workers, seed=args.seed, out=args.out)


if __name__ == "__main__":
    main()

