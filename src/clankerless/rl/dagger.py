"""DAgger: fix the cloned policy's compounding errors by labelling its own states.

Behavior cloning only ever sees positions the expert reaches. As soon as the cloned
policy plays a move F would not, it is off that distribution, its error rate rises, and
the mistakes feed on themselves -- which is why 82.6% move accuracy only bought a 35%
win rate.

DAgger (Ross et al., 2011) closes that loop: the *policy* plays the games, so the states
come from its own distribution including its blunders, and at every one of those states
the *expert* is asked what it would have done. Those (state, expert action) pairs are
appended to the dataset and the policy is retrained on everything collected so far.

Run:  python -m src.clankerless.rl.dagger --iterations 3 --games 800
"""

import argparse
import multiprocessing as mp
import os
import time
from argparse import Namespace

import numpy as np

from catanatron import Color, Game, GameAccumulator
from catanatron.features import create_sample_vector, get_feature_ordering
from catanatron.gym.envs.action_space import get_action_array, to_action_space
from catanatron.models.player import RandomPlayer
from catanatron.players.value import ValueFunctionPlayer

from src.clankerless.agents.heuristic import HeuristicPlayer
from src.clankerless.rl import bc_eval, bc_train
from src.clankerless.rl.policy_player import PolicyPlayer

DATA = os.path.join(os.path.dirname(__file__), "bc_data.npz")
CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "bc.pt")

_POLICY_CACHE = {}


class DaggerRecorder(GameAccumulator):
    """Record what the expert *would* have played at every state the policy reaches.

    Unlike ExpertRecorder, the action actually taken is ignored: the policy drives the
    game so that the states come from its own distribution, and the label is the
    expert's answer at that same state. The expert must share the policy's color so it
    evaluates the board from the same perspective.

    :param target_color: the color the policy plays (labels are recorded for it only)
    :param features: feature ordering used to build the observation vector
    :param player_colors: colors in the game, for the action-index mapping
    :param map_type: Catan map type
    """

    def __init__(self, target_color, features, player_colors, map_type="BASE"):
        self.target_color = target_color
        self.features = features
        self.player_colors = player_colors
        self.map_type = map_type
        self.n_actions = len(get_action_array(player_colors, map_type))
        self.expert = ValueFunctionPlayer(target_color)
        self.obs, self.acts, self.masks = [], [], []
        self.agreements = []

    def before(self, game):
        self.obs, self.acts, self.masks = [], [], []
        self.agreements = []

    def step(self, game_before_action, action):
        if action.color != self.target_color:
            return
        playable = game_before_action.playable_actions
        if len(playable) <= 1:
            return

        expert_action = self.expert.decide(game_before_action, playable)
        self.agreements.append(float(expert_action == action))

        self.obs.append(create_sample_vector(game_before_action, self.target_color, self.features))
        self.acts.append(to_action_space(expert_action, self.player_colors, self.map_type))
        m = np.zeros(self.n_actions, dtype=bool)
        for a in playable:
            m[to_action_space(a, self.player_colors, self.map_type)] = True
        self.masks.append(m)


def _get_policy(checkpoint):
    """Return a per-process PolicyPlayer, built once and reused across games."""
    if checkpoint not in _POLICY_CACHE:
        _POLICY_CACHE[checkpoint] = PolicyPlayer(Color.BLUE, checkpoint, greedy=False)
    return _POLICY_CACHE[checkpoint]


def _make_opponent(seed):
    """Pick the RED opponent, weighted toward F (the opponent we are measured against)."""
    pick = seed % 10
    if pick < 7:
        return ValueFunctionPlayer(Color.RED)
    if pick < 9:
        return HeuristicPlayer(Color.RED)
    return RandomPlayer(Color.RED)


def _rollout_one(task):
    """Play one game with the policy as BLUE, labelling every BLUE state with F's move.

    :param task: tuple (checkpoint, seed)
    :return: tuple (obs, expert actions, masks, agreement rate)
    """
    checkpoint, seed = task
    features = get_feature_ordering(2, "BASE")
    players = [_get_policy(checkpoint), _make_opponent(seed)]
    if seed % 2:
        players.reverse()

    rec = DaggerRecorder(Color.BLUE, features, (Color.BLUE, Color.RED))
    Game(players, seed=seed).play(accumulators=[rec])
    agree = float(np.mean(rec.agreements)) if rec.agreements else float("nan")
    return (np.array(rec.obs, np.float32),
            np.array(rec.acts, np.int64),
            np.array(rec.masks, dtype=bool),
            agree)


def collect(checkpoint, n_games, workers=None, seed=0):
    """Roll out the current policy and label its states with the expert's moves.

    :param checkpoint: policy checkpoint that drives the games
    :param n_games: games to play
    :param workers: worker processes (default: all CPUs)
    :param seed: base seed
    :return: tuple (obs, actions, masks, mean agreement with the expert)
    """
    tasks = [(checkpoint, seed + i) for i in range(n_games)]
    n_workers = workers or mp.cpu_count()
    chunk = max(1, n_games // (n_workers * 4))

    with mp.Pool(processes=workers) as pool:
        results = pool.map(_rollout_one, tasks, chunksize=chunk)

    results = [r for r in results if len(r[1]) > 0]
    obs = np.concatenate([r[0] for r in results])
    acts = np.concatenate([r[1] for r in results])
    masks = np.concatenate([r[2] for r in results])
    agree = float(np.nanmean([r[3] for r in results]))

    assert masks[np.arange(len(acts)), acts].all(), "expert action not marked legal"
    return obs, acts, masks, agree


def aggregate(data_path, obs, acts, masks):
    """Append newly labelled states to the dataset on disk.

    DAgger trains on the union of everything collected so far, so earlier expert data is
    kept rather than replaced -- otherwise the policy forgets the states it already
    handles well.

    :param data_path: .npz to grow
    :param obs: new observations
    :param acts: new expert action indices
    :param masks: new legality masks
    :return: total sample count after aggregation
    """
    if os.path.exists(data_path):
        old = np.load(data_path)
        obs = np.concatenate([old["obs"], obs])
        acts = np.concatenate([old["actions"], acts])
        masks = np.concatenate([old["masks"], masks])
    np.savez_compressed(data_path, obs=obs, actions=acts, masks=masks)
    return len(acts)


def run(args):
    """Run the DAgger loop: collect, aggregate, retrain, evaluate.

    :param args: parsed CLI arguments
    """
    print(f"DAgger: {args.iterations} iterations x {args.games} games, "
          f"data={args.data}, checkpoint={args.checkpoint}\n")

    for it in range(1, args.iterations + 1):
        t0 = time.perf_counter()
        print(f"--- iteration {it}/{args.iterations} ---")

        obs, acts, masks, agree = collect(
            args.checkpoint, args.games, args.workers, seed=args.seed + it * 100_000
        )
        total = aggregate(args.data, obs, acts, masks)
        print(f"collected {len(acts):,} new samples (policy agreed with F {agree * 100:.1f}% "
              f"of the time) -> dataset {total:,} | {time.perf_counter() - t0:.0f}s")

        train_args = Namespace(
            data=args.data, save=args.checkpoint, epochs=args.epochs,
            batch_size=512, learning_rate=1e-3, hidden=args.hidden,
            val_frac=0.1, torch_threads=0, seed=args.seed,
        )
        bc_train.train(train_args)

        if args.eval_games:
            print(f"evaluating iteration {it}:")
            bc_eval.evaluate(args.checkpoint, "F", args.eval_games, args.workers)
        print()


def build_parser():
    p = argparse.ArgumentParser(description="DAgger refinement of a cloned policy.")
    p.add_argument("--iterations", type=int, default=3, help="collect/retrain rounds")
    p.add_argument("--games", type=int, default=800, help="policy rollouts per iteration")
    p.add_argument("--epochs", type=int, default=30, help="retraining epochs per iteration")
    p.add_argument("--data", default=DATA, help="dataset to grow (starts from the BC data)")
    p.add_argument("--checkpoint", default=CHECKPOINT, help="policy to refine, updated in place")
    p.add_argument("--eval-games", type=int, default=300, help="games vs F after each iteration (0 to skip)")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
