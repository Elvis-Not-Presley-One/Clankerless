from collections import Counter
from typing import Any

import math

def win_rates(counter : Counter[Any], agents, games_played) -> dict[str, float]:
    """
    This function will calculate the win rate of each agent from a given Counter of games

        i.e. if you play 1000 games with random agent against AB agent, the win rate would be ~100%
            or Counter({AB: 1000},....)

    :param counter: a Counter object with the name of the model then the total amount of games won
                    in this format {[name : games won],....}
    :param agents: list of agents names that was selected to run the test with
    :param games_played: total number of games played

    :return: {agent_name: win_rate_percent}
    """
    return {
        name : round(100 * counter.get(name, 0) / games_played, 2)
        for name, _ in agents
    }


def win_rate_ci(wins, games, z=1.96):
    """
    the win_rate_ci() function calculates the illusions 95% confidence interval (CI)
    and returns the upper and lower bound of the confidence interval

    :param wins: total number of wins given the candidate agent/bot
    :param games: the number of games played
    :param z: the students z number for 95%

    :return: the upper and lower bound of the CI
    """
    p = wins / games
    d = 1 + z*z/games
    c = (p + z*z/(2*games)) / d
    h = (z/d) * math.sqrt(p*(1-p)/games + z*z/(4*games*games))

    return c - h, c + h

## TODO Think about adding elo system or Trueskill rating from phase 1.4








