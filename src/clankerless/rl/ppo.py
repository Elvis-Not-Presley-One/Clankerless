"""From-scratch PPO with invalid-action masking for Catanatron.

A single-file, readable implementation of Proximal Policy Optimization (Schulman et
al., 2017) in the CleanRL style, specialized for the Catanatron Gym environment
(``catanatron/Catanatron-v0``). The one Catan-specific piece is **invalid-action
masking**: the action space is a fixed ``Discrete(332)`` but only a handful of actions
are legal each turn, so before forming the action distribution we set the logits of
illegal actions to a large negative number. Their probability becomes ~0 and no
gradient flows to them (Huang & Ontanon, 2020).

The agent always plays ``Color.BLUE``; the opponent(s) are fixed bots configured on the
env and auto-play their turns. Reward is sparse and terminal: +1 win, -1 loss, 0
otherwise. Because we always submit a legal (masked) action, the env's invalid-action
penalty is never triggered.

Run:  python -m src.clankerless.rl.ppo --opponent random --total-timesteps 500000
"""

import argparse
import os
import random
import time
from collections import deque

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

import catanatron.gym  # noqa: F401  (registers catanatron/Catanatron-v0)
from catanatron import Color
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.models.player import RandomPlayer
from catanatron.players.value import ValueFunctionPlayer

from src.clankerless.rl.reward import make_reward

ENV_ID = "catanatron/Catanatron-v0"
MASK_FILL = -1e8  # logit value for illegal actions (=> ~0 probability)


def make_opponent(name, color=Color.RED):
    """Build a fixed opponent bot by short name.

    :param name: one of 'random', 'F' (ValueFunctionPlayer), 'AB' (AlphaBetaPlayer)
    :param color: the opponent's color (must not be BLUE, which the agent uses)
    :return: an instantiated Catanatron Player
    """
    if name == "random":
        return RandomPlayer(color)
    if name == "F":
        return ValueFunctionPlayer(color)
    if name == "AB":
        return AlphaBetaPlayer(color)
    raise ValueError(f"unknown opponent: {name}")


def make_env(opponent, seed = None, representation = "vector", reward = None):
    """Create a Catanatron Gym env with a fixed opponent.

    :param opponent: opponent short name (see make_opponent)
    :param seed: optional reset seed
    :param representation: 'vector' (flat Box) or 'mixed' (board tensor + numeric)
    :param reward: reward callable (action, game, p0_color) -> float; None keeps the
        env's default sparse simple_reward

    :return: the constructed gymnasium environment
    """
    config = {"enemies": [make_opponent(opponent)], "representation": representation}
    if reward is not None:
        # Only set the key when overriding: the env does config.get("reward_function",
        # simple_reward), so an explicit None would replace the default with None.
        config["reward_function"] = reward

    env = gymnasium.make(ENV_ID, config=config, disable_env_checker=True)
    if seed is not None:
        env.reset(seed=seed)
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonally initialize a linear layer (CleanRL convention).

    :param layer: the nn.Linear to initialize
    :param std: gain for the orthogonal weight init
    :param bias_const: constant to fill the bias with

    :return: the same layer, initialized in place
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)

    return layer


class Agent(nn.Module):
    """Actor-critic MLP with a shared input LayerNorm.

    The critic estimates state value; the actor produces logits over the full discrete
    action space, which callers mask before sampling. Input LayerNorm keeps the
    mixed-scale Catan features in a stable range without external running statistics.

    :param obs_dim: observation vector length
    :param n_actions: size of the discrete action space
    :param hidden: hidden layer width
    """

    def __init__(self, obs_dim, n_actions, hidden=256):

        super().__init__()
        self.norm = nn.LayerNorm(obs_dim)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, n_actions), std=0.01),
        )

    def get_value(self, x):
        """Return the critic's value estimate for observations ``x`` (shape [B, obs])."""
        return self.critic(self.norm(x))

    def get_action_and_value(self, x, mask, action=None, greedy=False):
        """Sample (or score) a masked action and return value.

        :param x: observations, shape [B, obs_dim]
        :param mask: bool legality mask, shape [B, n_actions] (True = legal)
        :param action: if given, score these actions instead of sampling
        :param greedy: if True and action is None, take the argmax legal action

        :return: (action, log_prob, entropy, value)
        """
        h = self.norm(x)
        logits = self.actor(h).masked_fill(~mask, MASK_FILL)
        dist = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if greedy else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(h)

    @classmethod
    def from_checkpoint(cls, checkpoint, device=None):
        """Rebuild an Agent matching a saved checkpoint's architecture, then load it.

        The observation size, action count, and hidden width are all inferred from the
        saved weight shapes, so the reconstructed network always matches the checkpoint
        even if it was trained at a different width -- callers never have to remember
        the hyperparameters. Accepts a state_dict or a path to one.

        :param checkpoint: a state_dict, or a filesystem path to a torch.save'd one
        :param device: torch device to map onto (default: cpu)
        :return: an Agent in eval mode with the checkpoint weights loaded
        """
        device = device or torch.device("cpu")
        state = torch.load(checkpoint, map_location=device) if isinstance(checkpoint, str) else checkpoint
        obs_dim = state["actor.0.weight"].shape[1]
        hidden = state["actor.0.weight"].shape[0]
        last = max(int(k.split(".")[1]) for k in state if k.startswith("actor.") and k.endswith(".weight"))
        n_actions = state[f"actor.{last}.weight"].shape[0]

        agent = cls(obs_dim, n_actions, hidden).to(device)
        agent.load_state_dict(state)
        agent.eval()
        return agent


def evaluate(agent, opponent, n_games, device, base_seed=1_000_000):
    """Greedily play the agent vs a fixed opponent and return its win rate.

    :param agent: the trained Agent
    :param opponent: opponent short name
    :param n_games: number of evaluation games (seeded, held out from training)
    :param device: torch device
    :param base_seed: first eval seed (disjoint from training's unseeded games)

    :return: win rate in [0, 1]
    """
    env = make_env(opponent)
    base = env.unwrapped
    wins = 0
    for i in range(n_games):
        obs, _ = env.reset(seed=base_seed + i)
        done = False
        while not done:
            mask = torch.tensor(np.array(base.action_masks()), dtype=torch.bool, device=device).unsqueeze(0)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t, mask, greedy=True)
            obs, _, term, trunc, _ = env.step(int(action.item()))
            done = term or trunc
        # Ask the game who won -- never infer it from the reward sign, which is only
        # +1/-1 under the sparse default and meaningless under shaped rewards.
        wins += 1 if base.game.winning_color() == base.p0.color else 0
    env.close()
    return wins / n_games


def train(args):
    """Run PPO training and periodically evaluate against F.

    :param args: parsed hyperparameters (see build_parser)
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(args.opponent, seed=args.seed, reward=make_reward(args.reward))
    base_env = env.unwrapped
    obs_dim = int(np.array(env.observation_space.shape).prod())
    n_actions = env.action_space.n

    agent = Agent(obs_dim, n_actions, args.hidden).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    T = args.num_steps
    obs = torch.zeros((T, obs_dim), device=device)
    masks = torch.zeros((T, n_actions), dtype=torch.bool, device=device)
    actions = torch.zeros(T, dtype=torch.long, device=device)
    logprobs = torch.zeros(T, device=device)
    rewards = torch.zeros(T, device=device)
    dones = torch.zeros(T, device=device)
    values = torch.zeros(T, device=device)

    global_step = 0
    start_time = time.time()
    ep_returns = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    ep_wins = deque(maxlen=100)

    obs_np, _ = env.reset(seed=args.seed)
    next_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    next_mask = torch.tensor(np.array(base_env.action_masks()), dtype=torch.bool, device=device)
    next_done = torch.zeros((), device=device)
    ep_return, ep_len = 0.0, 0

    num_updates = args.total_timesteps // T
    batch_inds = np.arange(T)
    minibatch_size = T // args.num_minibatches

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(T):
            global_step += 1
            obs[step] = next_obs
            masks[step] = next_mask
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs.unsqueeze(0), next_mask.unsqueeze(0)
                )
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            obs_np, reward, term, trunc, _ = env.step(int(action.item()))
            rewards[step] = float(reward)
            next_done = torch.tensor(float(term or trunc), device=device)
            ep_return += float(reward)
            ep_len += 1

            if term or trunc:
                # Record the winner BEFORE reset() -- reset builds a new Game and the
                # old result is gone. Win rate must come from the game, not the reward.
                ep_wins.append(1.0 if base_env.game.winning_color() == base_env.p0.color else 0.0)
                ep_returns.append(ep_return)
                ep_lengths.append(ep_len)
                ep_return, ep_len = 0.0, 0
                obs_np, _ = env.reset()

            next_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
            next_mask = torch.tensor(np.array(base_env.action_masks()), dtype=torch.bool, device=device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).flatten()
            advantages = torch.zeros(T, device=device)
            lastgaelam = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        clipfracs = []
        for _ in range(args.update_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, T, minibatch_size):
                mb = batch_inds[start:start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs[mb], masks[mb], actions[mb]
                )
                logratio = newlogprob - logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        mean_ret = np.mean(ep_returns) if ep_returns else float("nan")
        win_rate = np.mean(ep_wins) if ep_wins else float("nan")
        print(
            f"upd {update:4d}/{num_updates} | step {global_step:7d} | "
            f"train_win {win_rate * 100:5.1f}% | ret {mean_ret:+.3f} | "
            f"len {np.mean(ep_lengths) if ep_lengths else float('nan'):4.0f} | "
            f"vloss {v_loss.item():.3f} | pgloss {pg_loss.item():+.3f} | "
            f"ent {entropy_loss.item():.3f} | kl {approx_kl.item():.4f} | {sps} sps"
        )

        if args.eval_every and update % args.eval_every == 0:
            wr = evaluate(agent, args.eval_opponent, args.eval_games, device)
            print(f"    [eval] greedy vs {args.eval_opponent}: {wr * 100:.1f}%  ({args.eval_games} games)")

    env.close()
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        torch.save(agent.state_dict(), args.save)
        print("saved ->", args.save)
    return agent


def build_parser():
    """Build the CLI parser with PPO hyperparameters."""
    p = argparse.ArgumentParser(description="Masked PPO for Catanatron.")
    p.add_argument("--opponent", default="random", choices=["random", "F", "AB"], help="training opponent")
    p.add_argument("--reward", default="simple", choices=["simple", "vp"],
                   help="'simple' = sparse win/loss (env default), 'vp' = dense victory-point shaping")
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--num-steps", type=int, default=2048, help="rollout length per update")
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--norm-adv", action="store_true", default=True)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--cuda", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10, help="evaluate every N updates (0 to disable)")
    p.add_argument("--eval-opponent", default="F", choices=["random", "F", "AB"])
    p.add_argument("--eval-games", type=int, default=100)
    p.add_argument("--save", default="src/clankerless/rl/checkpoints/ppo.pt", help="checkpoint path ('' to skip)")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
