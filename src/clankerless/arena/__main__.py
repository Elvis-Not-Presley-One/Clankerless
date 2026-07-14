import argparse
from collections import Counter

from src.clankerless.agents import AGENTS
from src.clankerless.arena.arena import arena_play
from rich.console import Console
from rich.table import Table

from src.clankerless.arena.stats import win_rates, win_rate_ci


def print_results(counter, agents, games_played):
    """
    this function prints out the results of the test in a table for better visualization and
    so you don't need to bleach you eyes

    :param counter: the counter of the test results
    :param agents: list of all agents used in the test
    :param games_played: the number of games played

    :return: a pretty table in the console displaying all the results
    """
    rates = win_rates(counter, agents, games_played)
    table = Table(title=f"Arena Play Results - {games_played} games")
    table.add_column("Agent", style='cyan', no_wrap=True)
    table.add_column("Wins", justify='right')
    table.add_column("Win Rate", justify='right')
    table.add_column("95% CI", justify='right')

    for name in sorted(rates, key=rates.get, reverse=True):
        wins = counter.get(name, 0)
        lo, hi = win_rate_ci(wins, games_played)
        table.add_row(name, str(wins), f"{rates[name]:.1f}%",
                      f"[{100 * lo:.1f}, {100 * hi:.1f}]")

    draws = counter.get('DRAW', 0)
    table.add_section()
    table.add_row("Draws (Turn Limits)", str(draws),
                  f"{100 * draws / games_played:.1f}%", "", style='dim')

    Console().print(table)

def main():
    """
    This is the cli for the program specifically for the arena. The arena is responsible for handling all
    things
    :return:
    """
    p = argparse.ArgumentParser()
    p.add_argument('--agents', default = 'random,AB')
    p.add_argument('--games', type = int, default = 1000)
    args = p.parse_args()

    names = args.agents.split(',')
    agents = [(n, AGENTS[n]) for n in names]
    counter = arena_play(agents, args.games)
    print_results(counter, agents, args.games)

if __name__ == '__main__':
    main()
