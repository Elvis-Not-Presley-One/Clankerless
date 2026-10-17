"""AlphaZero-style MCTS for Catan: policy priors + real search.

Everything measured so far says the missing ingredient is search, not a better network.
F is itself a 1-ply search over base_fn; a network trained to imitate it reaches ~86%
move accuracy and only ~36% win rate, because an approximation of a searcher loses to
the searcher. AlphaBeta -- the same evaluation with 2-ply search -- beats F outright.

This combines the pieces we have:

* **Policy prior (PUCT)** from the behavior-cloned network, so search effort goes to the
  handful of plausible moves instead of spreading over ~50 legal ones. The prior is only
  applied at our own nodes: the action space encodes absolute robber victims, so the
  BLUE-trained policy cannot score an opponent's moves.
* **base_fn at the leaves** instead of the value head or random playouts. Benchmarks put
  base_fn at ~5.6k evals/s against ~670/s for a batch-1 network forward, and base_fn is
  the evaluation F itself uses -- ten times cheaper and stronger.
* **Chance handled by sampling** ``execute_spectrum``, which already enumerates dice and
  development-card outcomes with their probabilities.

Values are always stored from the searching player's perspective. Catan turns are not
strictly alternating (a player rolls, builds, then ends turn), so instead of negating
each ply we check whose turn a node belongs to and let the opponent minimise.
"""

import math
import random

import numpy as np
import torch

from catanatron import Color, Game, Player
from catanatron.features import create_sample_vector, get_feature_ordering
from catanatron.gym.envs.action_space import get_action_array, to_action_space
from catanatron.players.tree_search_utils import execute_spectrum, list_prunned_actions
from catanatron.players.value import base_fn, DEFAULT_WEIGHTS
from catanatron.state_functions import get_actual_victory_points

from src.clankerless.rl.ppo import Agent

# base_fn is dominated by its public_vps term (3e14), so squashing the raw difference
# flattens every position that is level on victory points -- a ~1e8 production edge
# divided by 1e14 is indistinguishable from zero. The two scales are therefore handled
# separately: victory points explicitly, and everything else through a copy of base_fn
# with public_vps zeroed.
# Scales measured rather than guessed: across mid-game states the positional difference
# between players has a median of ~2.8e7 (p90 ~8.8e7) and the victory-point difference a
# median of 0 (p90 3). A scale far above those flattens the value function to zero, which
# is what a first pass at 5e8 did -- the search then just followed the policy prior.
VP_SCALE = 3.0        # victory-point lead that maps to tanh(1)
POSITION_SCALE = 5e7  # positional base_fn difference that maps to tanh(1)
VP_WEIGHT = 0.6       # split between "who is ahead" and "who is better placed"

# The cloned policy is extremely confident -- cross-entropy training routinely puts
# ~0.99 on one action. Left alone, PUCT then sends every simulation down that branch and
# the search degenerates into the policy it was meant to improve on. Softening the prior
# and adding Dirichlet noise at the root (as AlphaZero does) restores exploration.
PRIOR_TEMP = 2.0
DIRICHLET_EPS = 0.25
DIRICHLET_ALPHA = 0.3


class Node:
    """One state in the search tree.

    Children are created lazily: expansion records the legal actions and their priors,
    but the successor game states are only built when a branch is first selected, which
    keeps a wide root from copying fifty games it may never visit.

    :param game: the game state at this node
    :param parent: parent Node, or None for the root
    :param prior: policy prior of the action that led here
    """

    __slots__ = ("game", "parent", "prior", "visits", "value_sum",
                 "actions", "priors", "children", "expanded")

    def __init__(self, game, parent=None, prior=0.0):
        self.game = game
        self.parent = parent
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.actions = []
        self.priors = []
        self.children = {}
        self.expanded = False

    @property
    def value(self):
        """Mean value of this node from the searching player's perspective."""
        return self.value_sum / self.visits if self.visits else 0.0

    def is_terminal(self):
        return self.game.winning_color() is not None


class MctsPlayer(Player):
    """Catan player that picks moves by policy-guided Monte Carlo tree search.

    :param color: the player's color (use Color.BLUE to match the trained policy)
    :param simulations: tree simulations per decision
    :param c_puct: exploration constant in the PUCT formula
    :param checkpoint: policy checkpoint for priors; None uses uniform priors
    :param leaf: leaf evaluator, 'base_fn' or 'value_net'
    :param prune: use Catanatron's action pruning to narrow the branching factor
    :param top_k: keep only the k highest-prior actions at our nodes (0 = keep all)
    :param is_bot: whether this player is computer-controlled
    """

    def __init__(self, color, simulations=100, c_puct=1.5, checkpoint=None,
                 leaf="base_fn", prune=True, top_k=8, is_bot=True,
                 vp_weight=VP_WEIGHT, position_scale=POSITION_SCALE,
                 prior_temp=PRIOR_TEMP, dirichlet_eps=DIRICHLET_EPS,
                 dirichlet_alpha=DIRICHLET_ALPHA):
        super().__init__(color, is_bot)
        self.simulations = int(simulations)
        self.c_puct = float(c_puct)
        self.leaf = leaf
        self.prune = bool(prune)
        self.top_k = int(top_k)
        self.vp_weight = float(vp_weight)
        self.position_scale = float(position_scale)
        self.prior_temp = float(prior_temp)
        self.dirichlet_eps = float(dirichlet_eps)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.map_type = "BASE"
        self.player_colors = (Color.BLUE, Color.RED)
        self.features = get_feature_ordering(2, self.map_type)
        self.n_actions = len(get_action_array(self.player_colors, self.map_type))
        # Positional evaluation only: victory points are scored separately so that a
        # level-on-points position is still ranked by production and development.
        self._position_fn = base_fn({**DEFAULT_WEIGHTS, "public_vps": 0})

        self.agent = None
        if checkpoint is not None:
            self.agent = Agent.from_checkpoint(checkpoint, torch.device("cpu"))

    # ---------------------------------------------------------------- priors

    def _priors(self, game, actions):
        """Prior probability for each action, from the policy network if available.

        Only our own decisions are scored by the network. Opponent nodes fall back to a
        uniform prior because the action indices encode absolute robber victims, so the
        BLUE-trained policy would misread a move made from RED's seat.

        :param game: state at the node being expanded
        :param actions: the legal actions to score
        :return: list of priors summing to 1
        """
        n = len(actions)
        if self.agent is None or game.state.current_color() != self.color:
            return [1.0 / n] * n

        obs = create_sample_vector(game, self.color, self.features)
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask = np.zeros(self.n_actions, dtype=bool)
        idx = [to_action_space(a, self.player_colors, self.map_type) for a in actions]
        mask[idx] = True
        m = torch.tensor(mask).unsqueeze(0)

        with torch.no_grad():
            logits = self.agent.actor(self.agent.norm(x)) / self.prior_temp
            logits = logits.masked_fill(~m, -1e8)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        p = np.array([probs[i] for i in idx], dtype=np.float64)
        total = p.sum()
        return (p / total).tolist() if total > 0 else [1.0 / n] * n

    # ------------------------------------------------------------ evaluation

    def _leaf_value(self, game):
        """Value of a leaf state in [-1, 1] from the searching player's perspective.

        A finished game is worth +/-1. Otherwise the victory-point lead and the
        positional advantage are squashed separately and blended, so that positions
        level on points are still separated by how well they are developed.

        :param game: the state to evaluate
        :return: value in [-1, 1]
        """
        winner = game.winning_color()
        if winner is not None:
            return 1.0 if winner == self.color else -1.0

        if self.leaf == "value_net" and self.agent is not None:
            obs = create_sample_vector(game, self.color, self.features)
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                return float(torch.tanh(self.agent.get_value(x)).item())

        others = [c for c in game.state.colors if c != self.color]
        vp_diff = (get_actual_victory_points(game.state, self.color)
                   - max(get_actual_victory_points(game.state, c) for c in others))
        pos_diff = (self._position_fn(game, self.color)
                    - max(self._position_fn(game, c) for c in others))
        return (self.vp_weight * math.tanh(vp_diff / VP_SCALE)
                + (1 - self.vp_weight) * math.tanh(pos_diff / self.position_scale))

    # --------------------------------------------------------------- search

    def _legal_actions(self, game):
        """Legal actions at a state, optionally pruned to the sensible ones."""
        if not self.prune:
            return list(game.playable_actions)
        pruned = list_prunned_actions(game)
        return pruned if pruned else list(game.playable_actions)

    def _expand(self, node, add_noise=False):
        """Record the actions and priors available at a node.

        :param node: the Node to expand
        :param add_noise: mix Dirichlet noise into the priors (root only), so a very
            confident policy cannot starve every alternative of visits
        """
        actions = self._legal_actions(node.game)
        priors = self._priors(node.game, actions)

        if add_noise and self.dirichlet_eps > 0 and len(actions) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            priors = [(1 - self.dirichlet_eps) * p + self.dirichlet_eps * n
                      for p, n in zip(priors, noise)]

        if self.top_k and len(actions) > self.top_k and node.game.state.current_color() == self.color:
            keep = sorted(range(len(actions)), key=lambda i: priors[i], reverse=True)[:self.top_k]
            actions = [actions[i] for i in keep]
            total = sum(priors[i] for i in keep) or 1.0
            priors = [priors[i] / total for i in keep]

        node.actions = actions
        node.priors = priors
        node.expanded = True

    def _select(self, node):
        """Pick a child by PUCT, creating its game state on first visit.

        The opponent is assumed to minimise our value, so their nodes negate Q rather
        than maximising it. Stochastic actions sample a successor from execute_spectrum
        the first time the branch is taken.

        :param node: an expanded, non-terminal Node
        :return: tuple (action index, child Node)
        """
        ours = node.game.state.current_color() == self.color
        sqrt_n = math.sqrt(max(1, node.visits))

        best, best_score = None, -float("inf")
        for i, action in enumerate(node.actions):
            child = node.children.get(i)
            q = 0.0 if child is None or child.visits == 0 else child.value
            if not ours:
                q = -q
            score = q + self.c_puct * node.priors[i] * sqrt_n / (1 + (child.visits if child else 0))
            if score > best_score:
                best, best_score = i, score

        child = node.children.get(best)
        if child is None:
            outcomes = execute_spectrum(node.game, node.actions[best])
            states = [s for s, _ in outcomes]
            probs = [p for _, p in outcomes]
            state = random.choices(states, weights=probs, k=1)[0] if len(states) > 1 else states[0]
            child = Node(state, parent=node, prior=node.priors[best])
            node.children[best] = child
        return best, child

    def _simulate(self, root):
        """Run one selection/expansion/evaluation/backup pass from the root."""
        node = root
        path = [node]

        while node.expanded and not node.is_terminal():
            _, node = self._select(node)
            path.append(node)

        if not node.is_terminal():
            self._expand(node)
        value = self._leaf_value(node.game)

        for n in path:
            n.visits += 1
            n.value_sum += value

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        root = Node(game.copy())
        self._expand(root, add_noise=True)
        if not root.actions:
            return playable_actions[0]

        for _ in range(self.simulations):
            self._simulate(root)

        # Visit count is the robust choice: it reflects where search actually spent its
        # effort, rather than a mean that a single lucky rollout can inflate.
        best = max(range(len(root.actions)),
                   key=lambda i: (root.children[i].visits if i in root.children else 0))
        chosen = root.actions[best]
        return chosen if chosen in playable_actions else playable_actions[0]

    def __repr__(self):
        prior = "net" if self.agent is not None else "uniform"
        return f"{type(self).__name__}:{self.color.value}({self.simulations}sim,{prior},{self.leaf})"
