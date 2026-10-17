"""Vectorized masked PPO: the same algorithm as ppo.py, stepping many games at once.

The single-env trainer in ppo.py is bound by simulator speed (~85 steps/s vs F), which
leaves most cores idle. This runs ``--num-envs`` copies of the environment in parallel
subprocesses via gymnasium's AsyncVectorEnv, so each network forward pass decides moves
for every game at once. The learning algorithm is unchanged -- only the plumbing gains
an environment dimension.

Everything else (Agent, evaluate, the reward factory) is imported from ppo.py, so the
two trainers stay in sync and can be compared directly.

Note that ``--num-steps`` is per environment: the batch each update trains on is
``num_steps * num_envs``.

Run:  python -m src.clankerless.rl.ppo_vec --opponent F --reward vp --num-envs 16 --num-steps 128
"""

import os
import random
import time
from collections import deque

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.clankerless.rl.ppo import Agent, build_parser, evaluate
from src.clankerless.rl.vec_env import EnvFactory


def masks_from_infos(infos, num_envs, n_actions):
    """Build the batched legality mask from a vector env's info dict.

    :param infos: the info dict returned by a vectorized env (has 'valid_actions')
    :param num_envs: number of parallel environments
    :param n_actions: size of the discrete action space

    :return: bool array of shape [num_envs, n_actions], True where an action is legal
    """
    mask = np.zeros((num_envs, n_actions), dtype=bool)
    for j in range(num_envs):
        mask[j, infos["valid_actions"][j]] = True
    return mask


def set_policy_trainable(agent, trainable):
    """Freeze or unfreeze everything that affects the policy.

    Behavior cloning trains only the actor, so a fine-tune starts with a randomly
    initialized critic whose advantage estimates are noise. Updating the policy against
    that noise destroys the cloned weights, so the critic is trained alone first. The
    input LayerNorm is frozen alongside the actor: it is shared, and letting the critic
    reshape it would silently change the policy's inputs.

    :param agent: the Agent to modify
    :param trainable: False to freeze the actor and norm, True to unfreeze
    """
    for p in agent.actor.parameters():
        p.requires_grad_(trainable)
    for p in agent.norm.parameters():
        p.requires_grad_(trainable)


def train_vec(args):
    """Run vectorized PPO training and periodically evaluate against a fixed opponent.

    :param args: parsed hyperparameters (see build_parser, plus --num-envs)

    :return: the trained Agent
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.torch_threads:
        # Env subprocesses already saturate the cores; letting torch spawn a thread per
        # core on top of that oversubscribes the CPU and slows the rollout down.
        torch.set_num_threads(args.torch_threads)

    N = args.num_envs
    envs = gymnasium.vector.AsyncVectorEnv(
        [EnvFactory(args.opponent, args.reward) for _ in range(N)]
    )
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    n_actions = int(envs.single_action_space.n)

    if getattr(args, "init_checkpoint", None):
        agent = Agent.from_checkpoint(args.init_checkpoint, device)
        agent.train()   # from_checkpoint leaves the module in eval mode
        print(f"initialized policy from {args.init_checkpoint}")
    else:
        agent = Agent(obs_dim, n_actions, args.hidden).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    warmup = getattr(args, "warmup_updates", 0)
    if warmup:
        set_policy_trainable(agent, False)
        print(f"critic-only warmup for {warmup} updates (policy frozen)")

    T = args.num_steps
    obs = torch.zeros((T, N, obs_dim), device=device)
    masks = torch.zeros((T, N, n_actions), dtype=torch.bool, device=device)
    actions = torch.zeros((T, N), dtype=torch.long, device=device)
    logprobs = torch.zeros((T, N), device=device)
    rewards = torch.zeros((T, N), device=device)
    dones = torch.zeros((T, N), device=device)
    values = torch.zeros((T, N), device=device)

    global_step = 0
    start_time = time.time()
    ep_returns = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    ep_wins = deque(maxlen=100)

    obs_np, infos = envs.reset(seed=args.seed)
    next_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    next_mask = torch.tensor(masks_from_infos(infos, N, n_actions), device=device)
    next_done = torch.zeros(N, device=device)
    ep_return = np.zeros(N)
    ep_len = np.zeros(N, dtype=int)

    best_eval = -1.0
    batch_size = T * N
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size
    batch_inds = np.arange(batch_size)

    print(f"vectorized PPO: {N} envs x {T} steps = {batch_size} batch, {num_updates} updates "
          f"(~{num_updates * batch_size:,} steps) vs {args.opponent}, reward={args.reward}")

    for update in range(1, num_updates + 1):
        warming = update <= warmup
        if warming:
            optimizer.param_groups[0]["lr"] = getattr(args, "warmup_lr", args.learning_rate)
        else:
            if warmup and update == warmup + 1:
                set_policy_trainable(agent, True)
                print(f"--- warmup done, unfreezing policy at lr {args.learning_rate:.1e} ---")
            if args.anneal_lr:
                frac = 1.0 - (update - warmup - 1.0) / max(1, num_updates - warmup)
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            else:
                optimizer.param_groups[0]["lr"] = args.learning_rate

        for step in range(T):
            global_step += N
            obs[step] = next_obs
            masks[step] = next_mask
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, next_mask)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            obs_np, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            done = np.logical_or(term, trunc)
            next_done = torch.tensor(done.astype(np.float32), device=device)

            ep_return += reward
            ep_len += 1
            if done.any():
                # The vector env has already auto-reset; the winner survives only in
                # final_info, stamped there by the WinnerInfo wrapper.
                final_info = infos.get("final_info")
                for j in np.nonzero(done)[0]:
                    info_j = final_info[j] if final_info is not None else None
                    if info_j is not None and "is_win" in info_j:
                        ep_wins.append(info_j["is_win"])
                    ep_returns.append(ep_return[j])
                    ep_lengths.append(ep_len[j])
                    ep_return[j] = 0.0
                    ep_len[j] = 0

            next_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
            next_mask = torch.tensor(masks_from_infos(infos, N, n_actions), device=device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)
            advantages = torch.zeros((T, N), device=device)
            lastgaelam = torch.zeros(N, device=device)
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

        b_obs = obs.reshape(-1, obs_dim)
        b_masks = masks.reshape(-1, n_actions)
        b_actions = actions.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        for _ in range(args.update_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = batch_inds[start:start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_masks[mb], b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_adv = b_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()
                if warming:
                    # Policy terms are excluded entirely so only the critic learns.
                    loss = args.vf_coef * v_loss
                else:
                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        mean_ret = np.mean(ep_returns) if ep_returns else float("nan")
        win_rate = np.mean(ep_wins) if ep_wins else float("nan")
        print(
            f"{'WARM ' if warming else 'upd '}{update:4d}/{num_updates} | step {global_step:8d} | "
            f"train_win {win_rate * 100:5.1f}% | ret {mean_ret:+.3f} | "
            f"len {np.mean(ep_lengths) if ep_lengths else float('nan'):4.0f} | "
            f"vloss {v_loss.item():.3f} | pgloss {pg_loss.item():+.3f} | "
            f"ent {entropy_loss.item():.3f} | kl {approx_kl.item():.4f} | {sps} sps"
        )

        if args.eval_every and update % args.eval_every == 0:
            wr = evaluate(agent, args.eval_opponent, args.eval_games, device)
            flag = ""
            # Fine-tuning can degrade a good starting policy, so keep the best snapshot
            # rather than whatever the final update happens to leave behind.
            if args.save and wr > best_eval:
                best_eval = wr
                os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                torch.save(agent.state_dict(), args.save)
                flag = "  <- saved"
            print(f"    [eval] greedy vs {args.eval_opponent}: {wr * 100:.1f}%  "
                  f"({args.eval_games} games){flag}")

    envs.close()
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        final = args.save if best_eval < 0 else args.save.replace(".pt", "_final.pt")
        torch.save(agent.state_dict(), final)
        print("saved ->", final)
        if best_eval >= 0:
            print(f"best eval checkpoint ({best_eval * 100:.1f}%) -> {args.save}")
    return agent


def build_vec_parser():
    """Extend the ppo.py parser with vectorization options."""
    p = build_parser()
    p.description = "Vectorized masked PPO for Catanatron."
    p.add_argument("--num-envs", type=int, default=16, help="parallel environments")
    p.add_argument("--init-checkpoint", default=None,
                   help="start from these weights (e.g. a behavior-cloned policy) instead of random init")
    p.add_argument("--warmup-updates", type=int, default=0,
                   help="updates to train the critic alone before unfreezing the policy; "
                        "required when starting from a BC checkpoint (its critic is untrained)")
    p.add_argument("--warmup-lr", type=float, default=1e-3, help="learning rate during the critic warmup")
    p.add_argument("--torch-threads", type=int, default=4,
                   help="torch CPU threads; keep low to avoid oversubscribing alongside "
                        "the env subprocesses (0 = leave torch's default)")
    # Per-env rollout length: the batch is num_steps * num_envs, so the single-env
    # default of 2048 would build a 32k batch at 16 envs.
    p.set_defaults(num_steps=128)
    return p


if __name__ == "__main__":
    train_vec(build_vec_parser().parse_args())
