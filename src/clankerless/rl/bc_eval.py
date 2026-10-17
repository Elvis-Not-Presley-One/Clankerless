"""Phase D: measure a trained policy as a real Catan player.

Validation accuracy says whether the network learned to predict F's moves; it does not
say whether that translates into winning. This plays the checkpoint through
PolicyPlayer in real games against a fixed opponent, in parallel, and reports a Wilson
95% confidence interval -- the same standard the arena and the heuristic ablations used,
so the numbers are directly comparable.

The policy is always seated as Color.BLUE (the action space encodes absolute robber
victims), but the move order alternates so it is not permanently first or second.

Run:  python -m src.clankerless.rl.bc_eval --checkpoint src/clankerless/rl/checkpoints/bc.pt --games 500
"""

import argparse
import multiprocessing as mp
import time

from catanatron import Color, Game
from catanatron.models.player import RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer

from src.clankerless.agents.heuristic import HeuristicPlayer
from src.clankerless.arena.stats import win_rate_ci
from src.clankerless.rl.policy_player import PolicyPlayer

_POLICY_CACHE = {}


def _get_policy(checkpoint, greedy):
    """Return a per-process PolicyPlayer, built once and reused across games.

    Loading a checkpoint per game would dominate the runtime, so each worker keeps its
    own instance. The player holds no per-game state, so reuse is safe.

    :param checkpoint: path to the saved Agent state_dict
    :param greedy: whether to take the argmax action instead of sampling
    :return: a PolicyPlayer seated as Color.BLUE
    """
    key = (checkpoint, greedy)
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = PolicyPlayer(Color.BLUE, checkpoint, greedy=greedy)
    return _POLICY_CACHE[key]


def _make_opponent(name):
    """Build the RED opponent by short name."""
    if name == "random":
        return RandomPlayer(Color.RED)
    if name == "F":
        return ValueFunctionPlayer(Color.RED)
    if name == "AB":
        return AlphaBetaPlayer(Color.RED)
    if name == "H":
        return HeuristicPlayer(Color.RED)
    raise ValueError(f"unknown opponent: {name}")


def _play_one(task):
    """Play one game of the policy (BLUE) vs a fixed opponent (RED).

    :param task: tuple (checkpoint, greedy, opponent_name, seed)
    :return: 1.0 if the policy won, else 0.0
    """
    checkpoint, greedy, opponent, seed = task
    players = [_get_policy(checkpoint, greedy), _make_opponent(opponent)]
    if seed % 2:
        players.reverse()   # alternate move order; the policy stays BLUE
    return 1.0 if Game(players, seed=seed).play() == Color.BLUE else 0.0


def evaluate(checkpoint, opponent="F", n_games=500, workers=None, greedy=True, base_seed=7_000_000):
    """Play the checkpoint against an opponent and report the win rate with a CI.

    :param checkpoint: path to the saved Agent state_dict
    :param opponent: opponent short name ('F', 'AB', 'H', 'random')
    :param n_games: number of games
    :param workers: worker processes (default: all CPUs)
    :param greedy: argmax action selection if True, sampling if False
    :param base_seed: first game seed (held out from any training seeds)
    :return: tuple (wins, n_games, win_rate, ci_low, ci_high)
    """
    tasks = [(checkpoint, greedy, opponent, base_seed + i) for i in range(n_games)]
    n_workers = workers or mp.cpu_count()
    chunk = max(1, n_games // (n_workers * 4))

    t0 = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        results = pool.map(_play_one, tasks, chunksize=chunk)

    wins = int(sum(results))
    lo, hi = win_rate_ci(wins, n_games)
    dt = time.perf_counter() - t0
    print(f"{opponent:6s} | {wins}/{n_games} = {wins / n_games * 100:5.1f}%  "
          f"95% CI [{lo * 100:.1f}%, {hi * 100:.1f}%]  ({dt:.1f}s)")
    return wins, n_games, wins / n_games, lo, hi


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy in real games.")
    parser.add_argument("--checkpoint", default="src/clankerless/rl/checkpoints/bc.pt")
    parser.add_argument("--opponents", nargs="+", default=["random", "F"],
                        choices=["random", "F", "AB", "H"], help="opponents to face")
    parser.add_argument("--games", type=int, default=500, help="games per opponent")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sample", action="store_true",
                        help="sample actions instead of taking the argmax")
    args = parser.parse_args()

    print(f"evaluating {args.checkpoint}  ({'sampling' if args.sample else 'greedy'})\n")
    print(f"{'opp':6s} | result")
    print("-" * 56)
    for opponent in args.opponents:
        evaluate(args.checkpoint, opponent, args.games, args.workers, greedy=not args.sample)


if __name__ == "__main__":
    main()
