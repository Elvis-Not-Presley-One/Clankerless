import multiprocessing as mp
import random
from collections import Counter
from typing import Any

import numpy as np

from catanatron import Color, Game
from src.clankerless.agents import AGENTS

def arena_play(agents, number_of_games = 1000, seed = 0) -> Counter[Any]:
    """
    the arena_play function is meant to play as many games as possible using multiprocessing to speed games up
    since each game takes around ~.13 seconds to play.

    :param agents: a list of agents to play through can be a max size of 4 (i.e 4 players)
    :param number_of_games: the number of games to play; defaults to 1000
    :param seed: the seed for the games that are going to be played
    :return: a counter with a list of the amount of games each bot won

            Example Return:

    """

    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
    wins = Counter()

    for i in range(number_of_games):
        seating = agents[:]
        random.Random(seed + 1).shuffle(seating)

        players = [cls(colors[j]) for j, (_, cls) in enumerate(seating)]
        by_color = {colors[j]: name for j, (name, _) in enumerate(seating)}
        winner = Game(players, seed=seed + 1).play()
        wins[by_color.get(winner, 'DRAW')] += 1

    return wins

