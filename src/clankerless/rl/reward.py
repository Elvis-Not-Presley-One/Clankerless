
from catanatron import Color, Game
from catanatron.state_functions import get_actual_victory_points


class VpReward:
    """"
    class that overrides the current reward function to create a vp reward function so the model can
    learn against better models
    """
    def __init__(self, scale = 0.1, win_bonus = 1.0):
        """the constructor"""
        self.scale = scale
        self.win_bonus = win_bonus
        self.prev_game = None
        self.prev_vp = 0


    def __call__(self, action, game : Game, p0_color : Color):
        """
        calls the reward function calculation when the instance is called

        :param action: current player action
        :param game: current game state
        :param p0_color: current player color

        :return: a float that represents the reward
            identified as (my_vp − prev_my_vp) * scale + 1 for win or -1 for loss
        """

        my_vp = get_actual_victory_points(game.state, p0_color)

        if game is not self.prev_game:
            self.prev_game = game
            self.prev_vp = 0

        reward = (my_vp - self.prev_vp) * self.scale
        self.prev_vp = my_vp

        winner = game.winning_color()
        if winner is not None:
            reward += self.win_bonus if p0_color == winner else - self.win_bonus

        return reward


def make_reward(name):
    """Build a reward function by short name (used by the --reward CLI flag).

    :param name: 'simple' (env default, sparse win/loss) or 'vp' (dense VpReward)

    :return: a reward callable, or None to leave the env's default in place
    """
    if name == "simple":
        return None
    if name == "vp":
        return VpReward()
    raise ValueError(f"unknown reward: {name}")
