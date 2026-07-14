from catanatron.models.player import SimplePlayer, RandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer

AGENTS = {
    'random' : RandomPlayer,
    'simple' : SimplePlayer,
    'AB' : AlphaBetaPlayer,
    'MCT' : MCTSPlayer
}

