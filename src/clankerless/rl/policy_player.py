"""
OBJECTIVE

the objective of this class is to create a player that can be driven by the trained ppo policy network

    so we are turning a normal Player -> objects into a PPO player -> tensors; in other words we translate
    Catan objects into something the network can read, then translate its answer back into a Catan action

    we use the same framework as the normal Player class where we have:
        init   -> store the heavy, time-consuming computational work (build + load the network)
        decide -> the quick step called thousands of times per game (translate, run net, translate back)

The network was trained as Color.BLUE in a 1v1 game on the BASE map. Seat the PolicyPlayer as BLUE
against a single opponent for valid play: the action space encodes absolute robber victims, so a
different seat would misalign the learned policy.
"""
import numpy as np
import torch

from catanatron import Player, Color
from catanatron.features import create_sample_vector, get_feature_ordering
from catanatron.gym.envs.action_space import (
    from_action_space,
    get_action_array,
    to_action_space,
)

from src.clankerless.rl.ppo import Agent


class PolicyPlayer(Player):
    """Catanatron Player driven by a trained PPO policy network.

    :param color: the player's color (use Color.BLUE to match training)
    :param checkpoint: path to a saved Agent state_dict (see ppo.py)
    :param greedy: if True take the highest-probability legal action, else sample
    :param num_players: players the network was trained with (sets the feature vector)
    :param map_type: Catan map the network was trained on
    """

    def __init__(self, color, checkpoint, greedy=True, num_players=2, map_type="BASE"):
        super().__init__(color)

        self.greedy = greedy
        self.map_type = map_type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.features = get_feature_ordering(num_players, map_type)
        self.player_colors = (Color.BLUE, Color.RED)
        self.n_actions = len(get_action_array(self.player_colors, map_type))

        self.agent = Agent.from_checkpoint(checkpoint, self.device)
        assert self.agent.actor[0].in_features == len(self.features), "obs size != feature ordering"
        assert self.agent.actor[-1].out_features == self.n_actions, "net actions != action array"

    def decide(self, game, playable_actions):
        """Choose an action by running the policy network over the legal moves.

        :param game: the current game state (read-only)
        :param playable_actions: the legal actions this turn
        :return: one action from playable_actions
        """
        if len(playable_actions) == 1:
            return playable_actions[0]

        obs = create_sample_vector(game, self.color, self.features)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        mask = np.zeros(self.n_actions, dtype=bool)
        for action in playable_actions:
            mask[to_action_space(action, self.player_colors, self.map_type)] = True
        mask_t = torch.tensor(mask, device=self.device).unsqueeze(0)

        with torch.no_grad():
            index, _, _, _ = self.agent.get_action_and_value(obs_t, mask_t, greedy=self.greedy)

        chosen = from_action_space(int(index.item()), self.color, self.player_colors, self.map_type)
        assert chosen in playable_actions, "decoded action is not playable"
        return chosen
