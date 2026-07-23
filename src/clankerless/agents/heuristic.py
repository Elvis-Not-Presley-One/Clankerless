import math

from catanatron import Game, Color, RESOURCES, Player
from catanatron.features import port_distance_features, build_production_features
from catanatron.models.map import number_probability
from catanatron.models.decks import SETTLEMENT_COST_FREQDECK, CITY_COST_FREQDECK, ROAD_COST_FREQDECK, \
    DEVELOPMENT_CARD_COST_FREQDECK
from catanatron.models.enums import SETTLEMENT, CITY, ROAD_BUILDING, ROAD, KNIGHT
from catanatron.players.value import base_fn, DEFAULT_WEIGHTS
from catanatron.state_functions import player_key, get_actual_victory_points, get_longest_road_length, \
    get_player_buildings, get_longest_road_color, get_player_freqdeck, get_played_dev_cards

MY_WEIGHTS = {**DEFAULT_WEIGHTS, 'longest_road': 6, 'hand_synergy': 35.0}


def port_synergy(game : Game, p0_color : Color, prod=None, ports=None) -> float:
    """
    the port_synergy function is meant to calculate the synergy of the resource production spots the player owns
    compared to the port resources the player owns.

    :param game: the current game board
    :param p0_color: the player owns color
    :param prod: optional precomputed build_production_features dict (computed if None)
    :param ports: optional precomputed port_distance_features dict (computed if None)

    :return:
    """

    synergy = 0.0
    if prod is None:
        prod = build_production_features(True)(game, p0_color)
    if ports is None:
        ports = port_distance_features(game, p0_color)

    for r in RESOURCES:
        if ports[f'P0_HAS_{r}_PORT']:
            synergy += prod[f'EFFECTIVE_P0_{r}_PRODUCTION']

    if ports["P0_HAS_3:1_PORT"]:
        synergy += 0.5 * sum(prod[f"EFFECTIVE_P0_{r}_PRODUCTION"] for r in RESOURCES)

    return synergy


def robber_threat(game : Game, p0_color : Color, ports=None, synergy=None) -> float:
    """
    the robber_threat function determines the likelihood of being targeted by the robber from another player from
    looking at these different articles

        Robber Threat
        What's likelyHood of being Targeted By a Robber From another Player?
            How Many Roads/Settlements i.e. Production Value
            How Many Harbors do I have/ What's my port_synergy
            How Many Victory Points I have/Victory Point Diff from other players
            Number of knight cards previously played
            what's the longest road I have currently compared to the current longest road

    :param game: the current game board is
    :param p0_color: the color of the player
    :param ports: optional precomputed port_distance_features dict (computed if None)
    :param synergy: optional precomputed port_synergy value (computed if None)

    :return: a float int that represents the likelihood of being targeted by the robber from another player
    """

    if ports is None:
        ports = port_distance_features(game, p0_color)
    amt_ports = sum(ports[f'P0_HAS_{r}_PORT'] for r in RESOURCES)  
    if synergy is None:
        synergy = port_synergy(game, p0_color)

    my_vp = get_actual_victory_points(game.state, p0_color)
    others = [get_actual_victory_points(game.state, c)
              for c in game.state.colors if c != p0_color]
    vp_lead = my_vp - max(others, default=0)

    my_road = get_longest_road_length(game.state, p0_color)
    has_longest_road = get_longest_road_color(game.state) == p0_color

    num_settlements = len(get_player_buildings(game.state, p0_color, SETTLEMENT))
    num_cities = len(get_player_buildings(game.state, p0_color, CITY))
    num_roads = len(get_player_buildings(game.state, p0_color, ROAD))

    return (0.8 * amt_ports + 0.5 * synergy + 0.3 * vp_lead
            + 0.4 * my_road + 1.0 * has_longest_road
            + 0.4 * num_settlements + 0.2 * num_cities + 0.2 * num_roads)


def robber_vulnerability(game: Game, p0_color: Color) -> float:
    """
    Worst-case single-tile production the player would lose to a robber placement.

    base_fn scores TOTAL effective production but is blind to how concentrated it is.
    The robber blocks exactly one tile, so the real forward risk is the most production
    the player draws from any single tile. Subtracting this term rewards spreading
    production across hexes (robustness) -- signal base_fn does not capture. Uses total
    (robber-agnostic) production so it measures exposure regardless of where the robber
    currently sits.

    :param game: the current game board
    :param p0_color: the color of the player

    :return: the maximum expected pips the player draws from any one tile, or 0.0 if the
        player has no producing tiles
    """
    cmap = game.state.board.map
    per_tile = {}

    for mult, kind in ((1, SETTLEMENT), (2, CITY)):
        for node_id in get_player_buildings(game.state, p0_color, kind):
            for tile in cmap.adjacent_tiles[node_id]:
                if tile.resource is None:
                    continue
                per_tile[tile.id] = per_tile.get(tile.id, 0.0) + mult * number_probability(tile.number)

    return max(per_tile.values(), default=0.0)


def port_trade_value(game: Game, p0_color: Color, prod=None, ports=None) -> float:
    """
    Trade value of the player's ports: how well they let the player dump SURPLUS.

    The earlier port_synergy hurt because it re-added raw production that base_fn
    already scores (its 3:1 branch added half of TOTAL production). This instead
    rewards only a port that matches a resource the player OVER-produces -- the
    surplus above their mean production, which is what a 2:1/3:1 port actually
    converts. base_fn has no port awareness at all, so this is non-redundant.

    :param game: the current game board
    :param p0_color: the color of the player
    :param prod: optional precomputed build_production_features dict (computed if None)
    :param ports: optional precomputed port_distance_features dict (computed if None)

    :return: a non-negative trade-value score (0.0 if no useful ports)
    """
    if prod is None:
        prod = build_production_features(True)(game, p0_color)
    if ports is None:
        ports = port_distance_features(game, p0_color)

    prod_by_r = [prod[f"EFFECTIVE_P0_{r}_PRODUCTION"] for r in RESOURCES]
    mean_prod = sum(prod_by_r) / len(RESOURCES)

    value = 0.0
    for r, pr in zip(RESOURCES, prod_by_r):
        if ports[f"P0_HAS_{r}_PORT"]:
            value += max(0.0, pr - mean_prod)

    if ports["P0_HAS_3:1_PORT"]:
        best_surplus = max((pr - mean_prod for pr in prod_by_r), default=0.0)
        value += 0.5 * max(0.0, best_surplus)

    return value


def turns_to_afford(cost, hand, prod) -> float:
    """
    the turns_to_afford() function calc the amnt of turns needed to get the cost of a buildable
    gated from your slowest resource

        Example of cost:
            settlement = [1,1,1,1,0]
                .
                .
                .

    :param cost: the cost of the buildable in the form of vector [WOOD, BRICK, SHEEP, WHEAT, ORE]
    :param hand: a vector [WOOD, BRICK, SHEEP, WHEAT, ORE] of your hand
    :param prod: a vector [WOOD, BRICK, SHEEP, WHEAT, ORE] of your current production

    :return: a float with the amount of turns needed to get the cost of a buildable
    """
    turns = 0.0

    for i in range(len(cost)):
        short = max(cost[i] - hand[i], 0)

        if short <= 0:
            continue

        if prod[i] == 0:

            return math.inf

        turns = max(turns, short / prod[i])

    return turns

def distance_to_next_vp(game : Game, p0_color : Color, hand=None, prod=None) -> float:
    """
    the distance_to_next_vp() function calculates the distance or amount of turns needed to get the next vp

    urns can be seen as num cards short / prod rate

            so we ask given the current cards we have about how many turns will it take to be able to buy
            a settlement, road, city, dev card

    :param game: the current game board is
    :param p0_color: the color of the player
    :param hand: optional precomputed freqdeck for the player (computed if None)
    :param prod: optional precomputed production vector [W,B,S,Wh,O] (computed if None)

    :return: a floating point number representing the least number of turns needed to obtain a vp
    """

    if hand is None:
        hand = get_player_freqdeck(game.state, p0_color)
    if prod is None:
        sample = build_production_features(True)(game, p0_color)
        prod = [sample[f'EFFECTIVE_P0_{r}_PRODUCTION'] for r in RESOURCES]

    settlement = turns_to_afford(SETTLEMENT_COST_FREQDECK, hand, prod)
    city = turns_to_afford(CITY_COST_FREQDECK, hand, prod)

    return min(settlement, city)


def _contest_value(mine, best_opp, threshold, chase_cost) -> float:
    """
    the contest_value() function calculates the contest value between the players hand and 2nd best opponents hands,
        hands meaning number of roads or knights.

    :param mine: the bots current amount of knights or roads
    :param best_opp: the opponents current amount of knights or roads
    :param threshold: the number of said card needed to first obtain the vp cards for longest roads or largest army
    :param chase_cost: the amount of turns it takes to be able to chase the vp against the other opponent

    :return: a float representing the contest value
    """

    if mine >= threshold and mine > best_opp:

        return (mine - best_opp) / mine * 2

    if best_opp >= threshold and best_opp >= mine:

        return mine / best_opp * 2 * (1 / (1 + chase_cost))

    return 0.0

def army_road_contest(game: Game, p0_color: Color, hand=None, prod=None) -> float:
    """
    The array_road_contest() function finds a float representing the contest of largest army and longest road
    between players for vp's

    :param game: the current game board
    :param p0_color: the color of the player
    :param hand: optional precomputed freqdeck for the player (computed if None)
    :param prod: optional precomputed production vector [W,B,S,Wh,O] (computed if None)

    :return: a float representing the contest of largest army and longest road
    """

    if hand is None:
        hand = get_player_freqdeck(game.state, p0_color)
    if prod is None:
        sample = build_production_features(True)(game, p0_color)
        prod = [sample[f'EFFECTIVE_P0_{r}_PRODUCTION'] for r in RESOURCES]
    opps = [c for c in game.state.colors if c != p0_color]

    mk, ok = get_played_dev_cards(game.state, p0_color, KNIGHT), max(get_played_dev_cards(game.state, c, KNIGHT) for c in opps)
    army = _contest_value(mk, ok, 3, turns_to_afford(DEVELOPMENT_CARD_COST_FREQDECK, hand, prod))

    mr, orr = get_longest_road_length(game.state, p0_color), max(get_longest_road_length(game.state, c) for c in opps)
    road = _contest_value(mr, orr, 5, turns_to_afford(ROAD_COST_FREQDECK, hand, prod))

    return army + road


# Ablation (2026-07-22, 350 games/config vs F, see src/clankerless/tuning/ablate.py)
# found all four features neutral or harmful, so they default to 0 and HeuristicPlayer
# reduces exactly to F. Re-enable a feature only once it beats F on its own.
DEFAULT_FEATURE_WEIGHTS = {
    "W_PORT": 0.0,     # port_trade_value: neutral both signs (~48%, no gain)
    "W_DIST": 0.0,     # distance_to_next_vp: hurts both signs (dup of base_fn hand_synergy)
    "W_CONTEST": 0.0,  # army_road_contest: neutral/hurts (redundant w/ base_fn road/army)
    "W_ROBBER": 0.0,   # robber_vulnerability: neutral both signs (~46%, no gain)
}


def my_value_fn(game: Game, p0_color: Color, weights=None, base_weights=None) -> float:
    """Score a board for ``p0_color``: Catanatron's base value plus custom feature terms.

    This is the evaluation the 1-ply HeuristicPlayer maximizes. With every feature
    weight at (effectively) zero it reduces exactly to ``base_fn(DEFAULT_WEIGHTS)`` --
    i.e. to what the ``F`` baseline uses -- so any measured edge over F is attributable
    solely to the feature terms and their weights.

    :param game: the current game state (read-only)
    :param p0_color: the color being scored
    :param weights: dict of feature weights with keys W_PORT, W_DIST, W_CONTEST,
        W_ROBBER; defaults to DEFAULT_FEATURE_WEIGHTS
    :param base_weights: weights dict passed to Catanatron's base_fn; defaults to
        DEFAULT_WEIGHTS (the same weights F uses)

    :return: the scalar value of the board for p0_color
    """
    weights = DEFAULT_FEATURE_WEIGHTS if weights is None else weights
    base_weights = DEFAULT_WEIGHTS if base_weights is None else base_weights

    # Compute the two expensive feature extractors and the hand once, then share them
    # across all feature functions (each recomputed them independently otherwise).
    prod_feats = build_production_features(True)(game, p0_color)
    ports = port_distance_features(game, p0_color)
    hand = get_player_freqdeck(game.state, p0_color)
    prod = [prod_feats[f"EFFECTIVE_P0_{r}_PRODUCTION"] for r in RESOURCES]

    base = base_fn(base_weights)(game, p0_color)
    return (base
            + weights["W_PORT"] * port_trade_value(game, p0_color, prod_feats, ports)
            + weights["W_DIST"] * (1 / (1 + distance_to_next_vp(game, p0_color, hand, prod)))
            + weights["W_CONTEST"] * army_road_contest(game, p0_color, hand, prod)
            - weights["W_ROBBER"] * robber_vulnerability(game, p0_color))


class HeuristicPlayer(Player):
    """1-ply greedy player that picks the action maximizing ``my_value_fn``.

    Identical search to Catanatron's ValueFunctionPlayer (``F``): it simulates each
    legal action one step and keeps the highest-scoring resulting board. The only
    difference from F is the value function, so a win over F isolates the quality of
    ``my_value_fn`` and its weights.

    :param color: the player's color
    :param weights: feature weights forwarded to my_value_fn (default: tuned defaults)
    :param base_weights: base_fn weights forwarded to my_value_fn (default: F's weights)
    :param is_bot: whether this player is computer-controlled
    """

    def __init__(self, color, weights=None, base_weights=None, is_bot=True):
        super().__init__(color, is_bot)
        self.weights = DEFAULT_FEATURE_WEIGHTS if weights is None else weights
        self.base_weights = DEFAULT_WEIGHTS if base_weights is None else base_weights

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        best, best_val = playable_actions[0], float("-inf")
        for action in playable_actions:
            g = game.copy()
            g.execute(action)
            val = my_value_fn(g, self.color, self.weights, self.base_weights)
            if val > best_val:
                best, best_val = action, val
        return best