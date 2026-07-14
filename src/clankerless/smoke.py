from catanatron import Game, RandomPlayer, Color
from collections import Counter

"""This is a smoke test to make sure everything is working as intended."""

N = 1000

players = [RandomPlayer(c) for c in Color]

wins = Counter()

for i in range(N):
    winner = Game(players).play()
    wins[winner] += 1

print(f'Games Played {N}')

for color in Color:
    n = wins[color]

    print(f'{color.value:<6} {n:4d} {100*n/N:5.1f}%')

none = wins[None]

print(f'{"None":<6} {none:4d} {100*none/N:5.1f}%   (Turn Limit No Winers)')
