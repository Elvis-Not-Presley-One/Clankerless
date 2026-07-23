"""CLI entry point for the weight tuner.

Run with:  python -m src.clankerless.tuning [options]
"""

import argparse

from src.clankerless.tuning.tuner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="CMA-ES search for HeuristicPlayer feature weights.")
    parser.add_argument("--generations", type=int, default=15, help="CMA-ES generations")
    parser.add_argument("--games", type=int, default=64, help="games per candidate per generation")
    parser.add_argument("--validate", type=int, default=800, help="fresh games to validate the winner (0 to skip)")
    parser.add_argument("--workers", type=int, default=None, help="worker processes (default: all CPUs)")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    parser.add_argument("--sigma", type=float, default=1.5, help="initial CMA-ES step size (log10 units)")
    args = parser.parse_args()

    run(generations=args.generations, games_per_eval=args.games, validate_games=args.validate,
        workers=args.workers, seed=args.seed, sigma0=args.sigma)


if __name__ == "__main__":
    main()
