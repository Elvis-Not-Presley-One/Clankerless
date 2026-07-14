import sys

from src.clankerless.arena.arena import arena_play
from src.clankerless.arena.stats import win_rate_ci


def gate(candidate, reference, games=1000):
    """
    the regression gate to test bots performance before being commited
    :param candidate: the candidate agent/bot to test
    :param reference: the reference agent/bot being tested a pon as the baseline
    :param games: the number of games to test

    :return: 0 if the test passes commits to github, 1 if the test fails does not commit to github,
    """
    counter = arena_play([(candidate, reference), ('ref', reference)], games)
    _, lo, _ = win_rate_ci(counter['cand'], games)
    passed = lo > 0.50
    print(f'cand {counter["cand"]}/{games} CI-low {lo:.1%}  -> {"PASS" if passed else "FAIL"}')
    sys.exit(0 if passed else 1)
