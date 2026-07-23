from catanatron.models.player import SimplePlayer, RandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer, DEFAULT_WEIGHTS
from catanatron.players.weighted_random import WeightedRandomPlayer
from src.clankerless.agents.heuristic import HeuristicPlayer, MY_WEIGHTS

AGENTS = {
    'random' : RandomPlayer,
    'simple' : SimplePlayer,
    'AB' : AlphaBetaPlayer,
    'MCT' : MCTSPlayer,
    'weighted' : WeightedRandomPlayer,
    'F' : ValueFunctionPlayer,
    # FW stands for ValueFunctionalPlayer Custom Weighted
    # i.e. I spent time to get the best weights possible, this so to find a model to run against
    'FW' : lambda color : ValueFunctionPlayer(color, value_fn_builder_name='C', params=MY_WEIGHTS),
    'VP' : VictoryPointPlayer,
    'H' : HeuristicPlayer
}

