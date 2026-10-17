import gymnasium

import catanatron.gym  # noqa: F401  (registers catanatron/Catanatron-v0)


class WinnerInfo(gymnasium.Wrapper):
    """"
    stamp the winner into info at termination so it survives vector reset
    """

    def step(self, action):
        """
        a step function created to grab the winners info before the env.reset() occurs when running
        multiple games on different cores.

        :param action: the possible actions a player can take

        :return: the observation, reward, done, info from the env
        """

        observation, reward, done, trunc, info = self.env.step(action)

        if done or trunc:
            base = self.env.unwrapped
            winner = base.game.winning_color()
            info["winner"] = winner
            info['is_win'] = float(winner == base.p0.color)

        return observation, reward, done, trunc, info

class EnvFactory:
    """
    picklable callable that builds one wrapped env
    """

    def __init__(self, opponent, reward_name):
        self.opponent = opponent
        self.reward_name = reward_name


    def __call__(self):
        """
        On call of this env factory we will make a new env that copies over from the old env and return the winner info
        :return:
        """
        from src.clankerless.rl.ppo import make_env
        from src.clankerless.rl.reward import make_reward

        env = make_env(self.opponent, reward = make_reward(self.reward_name))

        return WinnerInfo(env)

