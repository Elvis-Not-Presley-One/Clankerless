"""Train the policy network to imitate F's moves (behavior cloning).

Plain supervised classification: given a board, predict which of the 332 actions F
chose. Illegal actions are masked out of the logits before the loss, so the network
spends its capacity choosing among *legal* moves rather than relearning the rules --
masking already guarantees legality at play time.

Only the actor (and the shared input LayerNorm) is trained here; the critic is left at
its initialization because there is no value target in the dataset. That matters if you
later fine-tune with PPO: run a critic-only warmup first, or the untrained critic's
garbage advantages will destroy the cloned policy.

Run:  python -m src.clankerless.rl.bc_train --epochs 30
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.clankerless.rl.ppo import Agent

DATA = os.path.join(os.path.dirname(__file__), "bc_data.npz")
CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "bc.pt")


def load_data(path, val_frac=0.1, seed=0):
    """Load the recorded dataset and split it into train and validation tensors.

    The split is over samples rather than games, which slightly leaks positions from the
    same game across the split; it is still a fair test of "does this generalize to
    unseen boards" for our purposes and keeps the code simple.

    :param path: .npz produced by bc_data.py
    :param val_frac: fraction held out for validation
    :param seed: shuffle seed
    :return: tuple (train, val) of (obs, actions, masks) tensor triples
    """
    d = np.load(path)
    obs, acts, masks = d["obs"], d["actions"], d["masks"]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(acts))
    obs, acts, masks = obs[order], acts[order], masks[order]

    n_val = int(len(acts) * val_frac)
    to_t = lambda o, a, m: (torch.tensor(o, dtype=torch.float32),
                            torch.tensor(a, dtype=torch.long),
                            torch.tensor(m, dtype=torch.bool))
    return to_t(obs[n_val:], acts[n_val:], masks[n_val:]), to_t(obs[:n_val], acts[:n_val], masks[:n_val])


def masked_logits(agent, x, mask):
    """Actor logits with illegal actions pushed to -1e8 (probability ~0)."""
    return agent.actor(agent.norm(x)).masked_fill(~mask, -1e8)


@torch.no_grad()
def accuracy(agent, data, batch=4096):
    """Top-1 accuracy: how often the network picks exactly the action F played.

    :param agent: the Agent being trained
    :param data: an (obs, actions, masks) tensor triple
    :param batch: evaluation batch size
    :return: accuracy in [0, 1]
    """
    obs, acts, masks = data
    correct = 0
    for i in range(0, len(acts), batch):
        logits = masked_logits(agent, obs[i:i + batch], masks[i:i + batch])
        correct += (logits.argmax(dim=-1) == acts[i:i + batch]).sum().item()
    return correct / len(acts)


def train(args):
    """Run behavior cloning and save the resulting policy.

    :param args: parsed CLI arguments
    :return: the trained Agent
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    (tr_obs, tr_acts, tr_masks), val = load_data(args.data, args.val_frac, args.seed)
    obs_dim, n_actions = tr_obs.shape[1], tr_masks.shape[1]
    print(f"train {len(tr_acts):,} samples | val {len(val[1]):,} | obs {obs_dim} -> {n_actions} actions")

    agent = Agent(obs_dim, n_actions, args.hidden)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate)

    n = len(tr_acts)
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        agent.train()
        order = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, args.batch_size):
            idx = order[i:i + args.batch_size]
            logits = masked_logits(agent, tr_obs[idx], tr_masks[idx])
            loss = F.cross_entropy(logits, tr_acts[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        agent.eval()
        tr_acc = accuracy(agent, (tr_obs, tr_acts, tr_masks))
        val_acc = accuracy(agent, val)
        flag = ""
        if val_acc > best_val:
            best_val = val_acc
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            torch.save(agent.state_dict(), args.save)
            flag = "  <- saved"
        print(f"epoch {epoch:3d}/{args.epochs} | loss {total_loss / n:.4f} | "
              f"train acc {tr_acc * 100:5.1f}% | val acc {val_acc * 100:5.1f}%{flag}")

    print(f"\nbest val accuracy: {best_val * 100:.1f}%")
    print("saved ->", args.save)
    return agent


def build_parser():
    p = argparse.ArgumentParser(description="Behavior cloning from F's recorded moves.")
    p.add_argument("--data", default=DATA, help="dataset .npz from bc_data.py")
    p.add_argument("--save", default=CHECKPOINT, help="checkpoint path")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256, help="must match the PPO Agent width")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--torch-threads", type=int, default=0, help="0 = torch default")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
