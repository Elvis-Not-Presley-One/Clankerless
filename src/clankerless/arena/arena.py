import multiprocessing as mp
import random
from collections import Counter
from typing import Any

from catanatron import Color, Game
from src.clankerless.agents import AGENTS

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def _play_one_game(args) -> str:
    """Play ONE game, return the winning agent's name (or 'DRAW').

    Runs in a worker process. args are pickled to get here, so we pass agent
    *names* (strings), NOT agent objects, and rebuild players from AGENTS here.
    (Factories/lambdas in AGENTS aren't picklable — they must never cross over.)
    """
    names, seed = args
    seating = list(names)
    random.Random(seed).shuffle(seating)
    players = [AGENTS[name](COLORS[j]) for j, name in enumerate(seating)]
    by_color = {COLORS[j]: name for j, name in enumerate(seating)}
    winner = Game(players, seed=seed).play()
    return by_color.get(winner, "DRAW")


def arena_play(agents, number_of_games=1000, seed=0, workers=None) -> Counter[Any]:
    names = [name for name, _ in agents]
    tasks = [(names, seed + i) for i in range(number_of_games)]
    chunk = max(1, number_of_games // ((workers or mp.cpu_count()) * 8))

    with mp.Pool(processes=workers) as pool:
        results = pool.map(_play_one_game, tasks, chunksize=chunk)
    return Counter(results)