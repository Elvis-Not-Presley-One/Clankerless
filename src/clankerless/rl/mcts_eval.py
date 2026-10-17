"""Compare MCTS configurations against a fixed opponent, in parallel, with CIs.

Same discipline as the feature ablations: one variable at a time, the same held-out
seeds for every configuration, and a Wilson interval on each result so a few lucky games
cannot be mistaken for progress. The raw policy is included as the reference line --
search is only worth its cost if it beats the network it is built on.

Run:  python -m src.clankerless.rl.mcts_eval --games 200
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
from src.clankerless.rl.mcts import MctsPlayer
from src.clankerless.rl.policy_player import PolicyPlayer

CHECKPOINT = "src/clankerless/rl/checkpoints/bc.pt"
_CACHE = {}


def _init_worker():
    """Pin each worker to a single torch thread.

    Every worker runs its own network, and torch defaults to one thread per core, so a
    20-process pool spawns hundreds of threads competing for 20 cores. The forward passes
    here are batch-1 and gain nothing from intra-op parallelism.
    """
    import torch
    torch.set_num_threads(1)


def _build(spec):
    """Build (and cache per process) the player described by a config spec.

    :param spec: tuple (kind, frozen kwargs items)
    :return: a Player seated as Color.BLUE
    """
    if spec not in _CACHE:
        kind, items = spec
        kwargs = dict(items)
        if kind == "policy":
            _CACHE[spec] = PolicyPlayer(Color.BLUE, **kwargs)
        else:
            _CACHE[spec] = MctsPlayer(Color.BLUE, **kwargs)
    return _CACHE[spec]


def _opponent(name):
    if name == "F":
        return ValueFunctionPlayer(Color.RED)
    if name == "AB":
        return AlphaBetaPlayer(Color.RED)
    if name == "H":
        return HeuristicPlayer(Color.RED)
    if name == "random":
        return RandomPlayer(Color.RED)
    raise ValueError(f"unknown opponent: {name}")


def _play(task):
    """Play one game of a configured player (BLUE) against an opponent (RED)."""
    spec, opponent, seed = task
    players = [_build(spec), _opponent(opponent)]
    if seed % 2:
        players.reverse()   # alternate move order; our player stays BLUE
    return 1.0 if Game(players, seed=seed).play() == Color.BLUE else 0.0


def run(configs, opponent="F", n_games=200, workers=None, base_seed=5_000_000):
    """Evaluate each configuration on the same seeds and print a table.

    :param configs: list of (label, spec) pairs
    :param opponent: opponent short name
    :param n_games: games per configuration
    :param workers: worker processes (default: all CPUs)
    :param base_seed: first seed, held out from all training
    """
    n_workers = workers or mp.cpu_count()
    print(f"{len(configs)} configs x {n_games} games vs {opponent} on {n_workers} workers\n")
    print(f"{'config':34s} {'win%':>6s}  {'95% CI':>14s}   time")
    print("-" * 72)

    with mp.Pool(processes=workers, initializer=_init_worker) as pool:
        for label, spec in configs:
            t0 = time.perf_counter()
            tasks = [(spec, opponent, base_seed + i) for i in range(n_games)]
            chunk = max(1, n_games // (n_workers * 4))
            results = pool.map(_play, tasks, chunksize=chunk)
            wins = int(sum(results))
            lo, hi = win_rate_ci(wins, n_games)
            print(f"{label:34s} {wins / n_games * 100:5.1f}%  "
                  f"[{lo * 100:4.1f}, {hi * 100:4.1f}]  {time.perf_counter() - t0:6.0f}s")


def default_configs(checkpoint):
    """Scaling sweep for the configuration that won the first pass.

    The first sweep established three things: the network prior is essential (5.5% on
    uniform priors against 39.5% with the net at 50 simulations), Catanatron's action
    pruning costs both strength and time (46.0% without it against 40.5% with), and 150
    simulations beat 50. Pruning is therefore off everywhere here, and the open question
    is how far simulation count and a wider top_k keep paying.

    :param checkpoint: policy checkpoint used for priors
    :return: list of (label, spec) pairs
    """
    ck = checkpoint
    base = (("checkpoint", ck), ("prune", False))
    return [
        ("policy only (reference)", ("policy", (("checkpoint", ck),))),
        ("mcts 150 no-prune", ("mcts", base + (("simulations", 150),))),
        ("mcts 300 no-prune", ("mcts", base + (("simulations", 300),))),
        ("mcts 300 no-prune top_k 16", ("mcts", base + (("simulations", 300), ("top_k", 16)))),
        ("mcts 600 no-prune", ("mcts", base + (("simulations", 600),))),
    ]


def main():
    p = argparse.ArgumentParser(description="Sweep MCTS configurations.")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--opponent", default="F", choices=["F", "AB", "H", "random"])
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()
    run(default_configs(args.checkpoint), args.opponent, args.games, args.workers)


if __name__ == "__main__":
    main()
