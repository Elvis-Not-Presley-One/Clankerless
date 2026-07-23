"""CMA-ES search over the HeuristicPlayer's feature weights.

The search runs in log-space (it optimizes log10 of each weight) because the useful
weights span many orders of magnitude, and evaluates every candidate with common
random numbers (all candidates in a generation play the same seeds) so that score
differences reflect the weights rather than dice luck.

Objective: maximize HeuristicPlayer's win rate in 1v1 games against Catanatron's
ValueFunctionPlayer (``F``). Because H and F search identically deep (1 ply), a win
rate above 50% means the tuned evaluation is genuinely stronger than F's base_fn.
"""

import json
import math
import multiprocessing as mp
import os
import random
import time

from catanatron import Color, Game
from catanatron.players.value import ValueFunctionPlayer

from src.clankerless.agents.heuristic import HeuristicPlayer

COLORS = [Color.RED, Color.BLUE]

# Weights being tuned and the log10 range the search explores.
# 10^3 = negligible next to base_fn's ~1e8 move resolution; 10^12 = dominates it.
PARAMS = ["W_PORT", "W_DIST", "W_CONTEST", "W_ROBBER"]
LOG_LOWER = 3.0
LOG_UPPER = 12.0
START = {"W_PORT": 1e6, "W_DIST": 1e6, "W_CONTEST": 1e7, "W_ROBBER": 1e7}


def vec_to_weights(x) -> dict:
    """Convert a log10 search vector into a feature-weights dict.

    :param x: sequence of log10 exponents, one per entry of PARAMS
    :return: dict mapping each PARAMS name to 10**exponent
    """
    return {p: 10.0 ** float(xi) for p, xi in zip(PARAMS, x)}


def _play_tuning_game(task) -> float:
    """Play one 1v1 game of HeuristicPlayer(weights) against F.

    Runs in a worker process, so the argument is picklable (a weights dict of floats
    plus an int seed). Seating is shuffled by the seed so H and F split first-move
    advantage across the seed set.

    :param task: tuple (weights_dict, seed)
    :return: 1.0 if the HeuristicPlayer won, else 0.0
    """
    weights, seed = task
    seating = ["H", "F"]
    random.Random(seed).shuffle(seating)
    players = [
        HeuristicPlayer(COLORS[j], weights) if who == "H" else ValueFunctionPlayer(COLORS[j])
        for j, who in enumerate(seating)
    ]
    winner = Game(players, seed=seed).play()
    by_color = {COLORS[j]: who for j, who in enumerate(seating)}
    return 1.0 if by_color.get(winner) == "H" else 0.0


def evaluate_population(pop, seeds, pool, chunksize) -> list:
    """Evaluate every candidate on the same seed set and return each one's win rate.

    All candidate x seed games are flattened into a single parallel map so every
    worker stays busy, then results are grouped back by candidate.

    :param pop: list of weights dicts (the generation's candidates)
    :param seeds: seeds every candidate is scored on (common random numbers)
    :param pool: an open multiprocessing Pool
    :param chunksize: map chunk size

    :return: list of win fractions in [0, 1], aligned with pop
    """
    tasks, owner = [], []
    for i, weights in enumerate(pop):
        for s in seeds:
            tasks.append((weights, s))
            owner.append(i)

    results = pool.map(_play_tuning_game, tasks, chunksize=chunksize)

    wins = [0.0] * len(pop)
    for i, r in zip(owner, results):
        wins[i] += r
    return [w / len(seeds) for w in wins]


def validate(weights, n_games, pool, chunksize, base_seed=900_000) -> tuple:
    """Re-measure a weights dict on fresh, held-out seeds with a Wilson 95% CI.

    Uses a seed range disjoint from training so the reported win rate is not the one
    the search optimized against.

    :param weights: the feature-weights dict to test
    :param n_games: number of validation games
    :param pool: an open multiprocessing Pool
    :param chunksize: map chunk size
    :param base_seed: first validation seed (kept away from training seeds)

    :return: tuple (wins, n_games, win_rate, ci_low, ci_high) as fractions
    """
    from src.clankerless.arena.stats import win_rate_ci

    tasks = [(weights, base_seed + i) for i in range(n_games)]
    results = pool.map(_play_tuning_game, tasks, chunksize=chunksize)
    wins = int(sum(results))
    lo, hi = win_rate_ci(wins, n_games)
    return wins, n_games, wins / n_games, lo, hi


def _fmt(weights) -> str:
    """Format a weights dict compactly for a progress line."""
    return "  ".join(f"{k}={weights[k]:.2e}" for k in PARAMS)


def _save(weights, win_rate) -> str:
    """Persist the best weights to best_weights.json next to this module."""
    path = os.path.join(os.path.dirname(__file__), "best_weights.json")
    with open(path, "w") as f:
        json.dump({"weights": weights, "train_win_rate": win_rate}, f, indent=2)
    return path


def run(generations=15, games_per_eval=64, validate_games=800,
        workers=None, seed=0, sigma0=1.5) -> tuple:
    """Run the CMA-ES search and validate the winner.

    :param generations: number of CMA-ES generations
    :param games_per_eval: games per candidate per generation (common random numbers)
    :param validate_games: fresh games used to validate the best weights (0 to skip)
    :param workers: worker process count (default: all CPUs)
    :param seed: base RNG seed for CMA-ES and game seeds
    :param sigma0: initial CMA-ES step size (in log10 units)
    :return: tuple (best_weights_dict, best_train_win_rate)
    """
    import cma

    x0 = [math.log10(START[p]) for p in PARAMS]
    opts = {
        "bounds": [[LOG_LOWER] * len(PARAMS), [LOG_UPPER] * len(PARAMS)],
        "seed": seed + 1,
        "verbose": -9,
        "verb_disp": 0,
        "verb_log": 0,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    popsize = es.popsize
    n_workers = workers or mp.cpu_count()
    chunk = max(1, (popsize * games_per_eval) // (n_workers * 4))

    total = popsize * games_per_eval * generations
    print(f"CMA-ES: {len(PARAMS)} params, popsize {popsize}, {games_per_eval} games/eval, "
          f"{generations} gens (~{total:,} games) on {n_workers} workers")
    print(f"tuning {PARAMS} vs F, log10 bounds [{LOG_LOWER}, {LOG_UPPER}]\n")

    best_wr, best_weights = -1.0, vec_to_weights(x0)
    t0 = time.perf_counter()

    with mp.Pool(processes=workers) as pool:
        for gen in range(generations):
            X = es.ask()
            pop = [vec_to_weights(x) for x in X]
            seeds = [seed + gen * 10_000 + i for i in range(games_per_eval)]

            wr = evaluate_population(pop, seeds, pool, chunk)
            es.tell(X, [-w for w in wr])   # CMA minimizes; maximize win rate => minimize -wr

            gi = max(range(len(wr)), key=lambda i: wr[i])
            if wr[gi] > best_wr:
                best_wr, best_weights = wr[gi], pop[gi]
            elapsed = time.perf_counter() - t0
            print(f"gen {gen:2d} | gen-best {wr[gi] * 100:5.1f}% | overall-best {best_wr * 100:5.1f}% "
                  f"| {elapsed:6.1f}s | {_fmt(best_weights)}")

        print("\nbest weights:", _fmt(best_weights))
        print("saved ->", _save(best_weights, best_wr))

        if validate_games > 0:
            print(f"\nvalidating best weights on {validate_games} FRESH games vs F ...")
            wins, n, v_wr, lo, hi = validate(best_weights, validate_games, pool, chunk)
            print(f"  H wins {wins}/{n} = {v_wr * 100:.1f}%   95% CI [{lo * 100:.1f}%, {hi * 100:.1f}%]")
            if lo > 0.5:
                verdict = "BEATS F (CI clears 50%)"
            elif hi < 0.5:
                verdict = "loses to F"
            else:
                verdict = "inconclusive (CI spans 50%)"
            print(f"  -> {verdict}")

    return best_weights, best_wr
