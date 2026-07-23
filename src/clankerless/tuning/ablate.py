"""One-feature-at-a-time ablation for the heuristic evaluation.

For each feature weight, this activates that weight alone (the other three set to 0)
at a few strengths and measures win rate vs F, so we can attribute credit/blame per
feature. An all-off control is included and must land near 50% (all weights 0 means
my_value_fn reduces exactly to base_fn, i.e. H == F).

Every config is scored on the SAME held-out seed set (common random numbers) so the
differences reflect the feature, not the dice. Run with:

    python -m src.clankerless.tuning.ablate [--games N] [--workers K]
"""

import argparse
import multiprocessing as mp

from src.clankerless.arena.stats import win_rate_ci
from src.clankerless.tuning.tuner import PARAMS, _play_tuning_game

OFF = {p: 0.0 for p in PARAMS}
STRENGTHS = (-1e8, -1e7, 1e7, 1e8)   # both signs: does the feature help added OR subtracted?
BASE_SEED = 500_000


def build_configs(strengths=STRENGTHS, only=None) -> list:
    """Build the ablation configs: an all-off control plus each feature alone.

    :param strengths: weight magnitudes to test each feature at
    :param only: if given, restrict to these feature names (subset of PARAMS)
    :return: list of (label, weights_dict) pairs
    """
    params = only if only else PARAMS
    configs = [("control (all off)", dict(OFF))]
    for p in params:
        for w in strengths:
            configs.append((f"{p} = {w:.0e}", {**OFF, p: w}))
    return configs


def _score(weights, seeds, pool, chunk) -> int:
    """Play the seed set for one config and return HeuristicPlayer's win count."""
    tasks = [(weights, s) for s in seeds]
    return int(sum(pool.map(_play_tuning_game, tasks, chunksize=chunk)))


def run(n_games=350, workers=None, base_seed=BASE_SEED, only=None) -> None:
    """Run the ablation and print a per-config win-rate table with Wilson CIs.

    :param n_games: games per config (shared seed set across all configs)
    :param workers: worker process count (default: all CPUs)
    :param base_seed: first held-out seed
    :param only: if given, restrict to these feature names (subset of PARAMS)
    """
    seeds = [base_seed + i for i in range(n_games)]
    configs = build_configs(only=only)
    n_workers = workers or mp.cpu_count()
    chunk = max(1, n_games // (n_workers * 4))

    print(f"ablation: {len(configs)} configs x {n_games} games vs F  (feature alone, others off)")
    print(f"{'config':22s} {'win%':>6s}  {'95% CI':>14s}  verdict")
    print("-" * 58)

    with mp.Pool(processes=workers) as pool:
        for label, weights in configs:
            wins = _score(weights, seeds, pool, chunk)
            wr = wins / n_games
            lo, hi = win_rate_ci(wins, n_games)
            if lo > 0.5:
                verdict = "HELPS"
            elif hi < 0.5:
                verdict = "hurts"
            else:
                verdict = "neutral"
            print(f"{label:22s} {wr * 100:5.1f}%  [{lo * 100:4.1f}, {hi * 100:4.1f}]  {verdict}")
            if label.startswith("control"):
                print("-" * 58)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-feature-at-a-time ablation vs F.")
    parser.add_argument("--games", type=int, default=350, help="games per config")
    parser.add_argument("--workers", type=int, default=None, help="worker processes (default: all CPUs)")
    parser.add_argument("--only", nargs="+", default=None, metavar="W_NAME",
                        help="restrict to these feature slots (e.g. --only W_ROBBER)")
    args = parser.parse_args()
    run(n_games=args.games, workers=args.workers, only=args.only)


if __name__ == "__main__":
    main()
