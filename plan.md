# Clankerless — Catan Bot Project Plan

> **Goal:** build an AI that plays online Catan on **colonist.io** and wins consistently. Stretch goals: climb the ranked ladder, and make a YouTube video about the build.
>
> **Approach:** deep RL self-play (AlphaZero-style) is the north star, reached by first building strong *classical* baselines. Everything sits on the **Catanatron** simulator.
>
> **This doc doubles as a curriculum** — it's structured so you learn the concepts as you build. See [How to use this plan](#how-to-use-this-plan).
>
> Last updated: 2026-07-07 (v8 — restructured as a learning-oriented curriculum; dense API tables moved to appendices)

---

## Table of contents
- [How to use this plan](#how-to-use-this-plan)
- [1. The big picture](#1-the-big-picture)
- [2. Key concepts & vocabulary](#2-key-concepts--vocabulary)
- [3. Tech stack & key decisions](#3-tech-stack--key-decisions)
- [4. Architecture: one interface, many brains](#4-architecture-one-interface-many-brains)
- [5. The roadmap (the phases)](#5-the-roadmap-the-phases)
  - [Phase 0 — Get fluent with the simulator](#phase-0--get-fluent-with-the-simulator-)
  - [Phase 1 — Baselines + the arena](#phase-1--baselines--the-arena-)
  - [Phase 2 — A strong classical bot (your L1)](#phase-2--a-strong-classical-bot-your-l1-)
  - [Phase 3 — First learning agent (RL + imitation)](#phase-3--first-learning-agent-rl--imitation-)
  - [Phase 4 — AlphaZero self-play (the headline)](#phase-4--alphazero-self-play-the-headline-)
  - [Phase 5 — The colonist.io live interface](#phase-5--the-colonistio-live-interface)
  - [Phase 6 — Bridge the brain + go live](#phase-6--bridge-the-brain--go-live-)
  - [Phase 7 — YouTube production](#phase-7--youtube-production-)
- [6. The colonist.io interface (deep dive)](#6-the-colonistio-interface-deep-dive)
- [7. Compute & ELSA](#7-compute--elsa)
- [8. Risks, ethics & ToS](#8-risks-ethics--tos)
- [9. Timeline](#9-timeline)
- [10. Environment setup (done)](#10-environment-setup-done)
- [Appendix A — Catanatron API map (verified)](#appendix-a--catanatron-api-map-verified)
- [Appendix B — Catanatron state & data model (verified)](#appendix-b--catanatron-state--data-model-verified)
- [Appendix C — Repo structure](#appendix-c--repo-structure)
- [Appendix D — Resources](#appendix-d--resources)

---

## How to use this plan

- **The roadmap (§5) is the spine.** Do the phases in order; each is a self-contained mini-project with a clear **Done when**. Don't start a phase until the previous one's "Done when" is met.
- **Every step has the same shape**, so you always know what you're getting:
  > **N. What to do.** — *the how* (exact files, classes, functions). **Learn:** the one concept this step teaches.

  If you already know the concept, skim the step. If "Learn:" names something unfamiliar, that's your cue to go study it before writing the code — this is the point of the project.
- **Tags:** `[import]` = use Catanatron's code as-is · `[write]` = new code you author · 💻 = runs on your PC · 🖥️ = runs on the ELSA cluster.
- **The golden rule:** *never reimplement what Catanatron already ships — import it.* Your code is only: the **arena**, the **agents that don't exist yet**, the **training loop**, and the **live bridge**. Everything else is a library call.
- **Dense API facts live in the appendices** (A = symbols, B = data model) so the roadmap stays readable. When a step cites `file:line`, the full context is in an appendix.

Paths like `models/player.py:37` are relative to the Catanatron **source root**: `catanatron/catanatron/catanatron/`. Paths like `agents/arena.py` are relative to *your* package `src/catanbot/` (Appendix C).

---

## 1. The big picture

### 1a. The goal, as a ladder of wins
Ship *something demonstrable* at every level so the project never feels stuck:

| Level | You have… | Why it matters |
|---|---|---|
| **L0** | a fast sim + an **arena** that reports win-rates with error bars | you can't improve what you can't measure |
| **L1** | a **classical bot** that beats Catanatron's best built-in (`AB:2`) >55% (4p) | a genuinely strong, shippable bot + the sparring partner RL needs |
| **L2** | a **learned agent** (neural net) that beats your L1 bot | proof the learning loop actually works |
| **L3** | the bot **playing full games on colonist.io** end-to-end | the unique, video-worthy engineering feat |
| **L4** | a **positive win-rate vs. real ranked humans** | the headline dream (riskiest; see §8) |

**Minimum bar for a great video: L1 + L3.** L4 is the dream but least in your control.

### 1b. The hard truth (so you set expectations right)
Two well-documented prior attempts:
- **Catanatron's author** tried DQN, cross-entropy RL, supervised learning, and pure MCTS. **All lost to a hand-crafted evaluation function + depth-2 alpha-beta search.** Pure MCTS was too slow; RL on random data too noisy.
- **settlers-rl** trained PPO for **~1 month on an RTX 3090** (~450M decisions) and still landed *"not close to a good human."* Adding search on top of the trained net helped. Their advice: **drop trading first** — it wrecks sample efficiency.

**What this means for you:**
1. The strongest *known* Catan AI is **search + a good evaluation function**, not raw RL — so you build that first (it's also your benchmark and your RL teacher).
2. The realistic ceiling is **AlphaZero = a neural net guided by search (MCTS)**. Search-on-top-of-a-policy is the pattern that works.
3. **Simulator speed is everything** for self-play → that's why we use Catanatron.
4. **Cut trading at first.** Catan without player-to-player trading is still a full, winnable game and *far* easier to learn.

This isn't a detour from your RL goal — it's the professional path *to* it. Every serious AlphaZero project starts with a fast sim and strong baselines.

### 1c. The plan of attack (why this order)
```
   fast sim + arena        strong search bot          RL self-play            live bridge
   (measure everything) ─▶  (the bar to beat)  ─▶   (beat the bar)   ─▶   (play colonist.io)
        L0                       L1                      L2                    L3 / L4
```
The **arena** comes first or you'll fool yourself about whether changes help. The **search bot** is never throwaway — it's your benchmark, your RL opponent, and your imitation-learning *teacher*. The **live bridge** is independent of which brain you plug in, so start it early (§6) — it's the riskiest unknown.

---

## 2. Key concepts & vocabulary

One-line definitions of terms used throughout. If one is new to you, study it when its phase arrives.

- **Agent** (a.k.a. *brain* / *bot*) — anything that, given a board and the legal moves, returns one move. In code: Catanatron's `Player.decide(game, playable_actions) -> Action`. Random, heuristic, search, and neural-net bots are all agents; they differ only in *how* they pick.
- **Baseline** — a deliberately weak agent (random, weighted-random) that every serious bot must beat. Your fixed zero-point.
- **Arena** — your harness that plays many games between agents and reports win-rates **with confidence intervals**. Your scoreboard.
- **Confidence interval (CI) / Elo** — CIs tell you whether a win-rate gap is *real* or dice noise; Elo turns many match results into a single comparable rating.
- **Observation / encoding** — the board turned into numbers a neural net can read (a vector or tensor). "Encoding" is the code that does state → numbers.
- **Action mask** — a boolean vector marking which of the fixed action slots are legal right now, so the policy never picks an illegal move.
- **Policy / value net** — a neural net with two heads: *policy* = a probability over actions ("what to do"), *value* = expected game outcome ("how good is this position").
- **PPO** — a standard, robust reinforcement-learning algorithm for training a policy from self-play reward.
- **Behavioral cloning (BC)** — supervised learning where the net imitates a stronger player's moves. Your RL "warm start."
- **MCTS (Monte-Carlo Tree Search)** — look-ahead by simulating many move sequences; in AlphaZero it's *guided* by the net's policy/value.
- **Self-play** — the agent improves by playing against copies of itself, generating its own training data.
- **Hidden information / determinization** — you can't see opponents' hand cards; *determinization* = sampling a plausible full hand so search can proceed, then averaging over samples.

---

## 3. Tech stack & key decisions

| Concern | Choice | Why |
|---|---|---|
| **Simulator** | **Catanatron** (`bcollazo/catanatron`, installed editable) | Fastest Python Catan sim, has a Gym interface + strong built-in bots. Sim speed is the RL bottleneck, so this beats slower academic repos. |
| **Language / DL** | **Python 3.11 + PyTorch** | Catanatron needs 3.11; PyTorch is the standard for custom AlphaZero/PPO. |
| **First RL** | **PPO** via **sb3-contrib** (`MaskablePPO`) → **CleanRL** later | sb3 gets a learning curve fast (masking built in); CleanRL's single-file PPO is easy to hack for the custom multi-head net. |
| **High ceiling** | **Custom AlphaZero** (MCTS + policy/value net + self-play) | The known-best pattern; worth building yourself for control + the video. |
| **Classical bot** | Catanatron's `AlphaBetaPlayer` + **your** evaluation function | Proven strongest-known approach; also your benchmark. |
| **Live interface** | **Read** state via WebSocket sniffing; **act** via browser automation (Playwright) | Reading is passive/safe; clicking like a human is more robust and less detectable than injecting messages (§6). |
| **Experiment tracking** | **Weights & Biases** (or TensorBoard) | Self-play runs are long; you must chart win-rate vs. baselines over time. |

**Reuse vs. build (the golden rule, made concrete):**

| Layer | Reuse or build? | What |
|---|---|---|
| RL algorithm (PPO / MCTS code) | **reuse** | sb3-contrib / CleanRL — never hand-roll PPO |
| Baseline & teacher bots | **reuse** | Catanatron's `RandomPlayer`, `WeightedRandomPlayer`, `AlphaBetaPlayer` |
| Pre-trained Catan weights | **skip** | none are strong enough to matter (the one public model is below good-human) |
| Your evaluation fn, encoding, net, training loop, live bridge | **build** | this is the actual project |

**The shortcut worth knowing:** don't train from random. **Behavioral-clone** the net on games from your Phase-2 bot, *then* run RL/self-play on top (the AlphaGo recipe). It skips the slow cold-start that made from-scratch attempts cost a month.

---

## 4. Architecture: one interface, many brains

```
                 ┌──────────────────────────────────────────────┐
                 │                 AGENT API                     │
                 │   decide(game, playable_actions) -> action    │   ← every brain implements this
                 └──────────────────────────────────────────────┘
                     ▲            ▲             ▲            ▲
              ┌──────┘      ┌─────┘        ┌────┘       ┌────┘
        ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌──────────────┐
        │ Random / │  │ Heuristic +│  │ PPO net  │  │ AlphaZero    │
        │ Weighted │  │ AlphaBeta  │  │ (policy) │  │ (MCTS + net) │
        └──────────┘  └────────────┘  └──────────┘  └──────────────┘
                            │  shared: encoding + action mask │
                            ▼                                 ▼
             ┌───────────────────────┐        ┌───────────────────────────┐
             │  TRAINING (Phase 3-4) │        │  ARENA / EVAL (Phase 1)    │
             │  self-play → buffer   │        │  many games → win% + Elo   │
             │  → learner (GPU)      │        │  → regression gate         │
             └───────────────────────┘        └───────────────────────────┘

                    ═══ the SAME agent + encoding plug into ═══

             ┌────────────────────────────────────────────────────┐
             │  LIVE BRIDGE (Phase 5-6):  colonist.io              │
             │  sniff WS → parse to internal state → encode →      │
             │  agent.decide() → click on canvas                  │
             └────────────────────────────────────────────────────┘
```

**The one design rule that makes this whole thing work:** every brain implements the *same* `decide()` and consumes the *same* encoding. So you develop and train entirely offline, then swap the trained brain into the live game with **zero** changes to the integration layer. This is why the encoding (Phase 3) must be pinned down exactly — the live bridge reuses it byte-for-byte.

---

## 5. The roadmap (the phases)

> Phases 0–4 are the AI. Phase 5 (the live interface) is independent — **start it in parallel once Phase 2 is underway**, since it's the riskiest unknown.

---

### Phase 0 — Get fluent with the simulator  💻

**What you're building:** a working env and a script that runs 1,000 games and prints win-rates.
**Why / learn:** every later phase reduces to *read a board → return a move*. Before anything clever, you must see how Catanatron represents a game and how to drive it. **Concept:** the agent–environment loop, and Catanatron's `Game` / `Player` / `Action` model.

**Build it — in order:**
1. **Confirm the environment works.** It's already installed (§10); just verify `catanatron-play --players=R,R,R,R --num=10` prints a results table. **Learn:** editable installs + venv isolation (why the bot code and your code share one interpreter).
2. **[write] `src/catanbot/smoke.py`.** Loop 1,000 games of four `RandomPlayer`s: `Game(players, seed=i).play()` returns the winning `Color`, or `None` on turn-limit. Tally with `collections.Counter` (handle `None` as its own bucket) and print win-rate per color. **Learn:** the driver loop — construct a game, step it to completion, read the result.
3. **[read] Click through the core source.** `models/player.py` (`decide`), `models/enums.py` (`ActionType`, the `Action` namedtuple), `game.py` (`Game.play`). Full map in **Appendix A**. **Learn:** the single method every agent implements — `decide(game, playable_actions) -> Action`.
4. **[read] Inspect a live state.** In a REPL: `g = Game([...]); print(g.state.player_state); print(g.playable_actions)`. Poke `g.state.board`, `g.state.resource_freqdeck`. Map in **Appendix B**. **Learn:** how hands/buildings/dev-cards are stored — you'll mirror this in every encoder.
5. **[optional] Watch a game.** Bring up Catanatron's web UI (Docker). **Learn:** nothing technical — but watching games builds the Catan intuition your evaluation function will need.

**Watch out:** four identical random bots should each win **~25%** — that's your sanity check. A color far off 25% means a bug (or you forgot seating is auto-shuffled).

**Done when:** `smoke.py` prints a win-rate table over 1,000 games and you can explain `decide()` in one sentence.
🎥 The first web-UI game.

---

### Phase 1 — Baselines + the arena  💻

**What you're building:** the **arena** — your measurement harness — plus a registry of agents to feed it.
**Why / learn:** you're building the *scoreboard before the players*. Every "is my new idea actually better?" question for the whole project gets answered here. **Concept:** rigorous evaluation under noise — sample sizes, confidence intervals, and ratings.

**Build it — in order:**
1. **[import] Your baselines already exist — do NOT rewrite them.** `from catanatron import RandomPlayer` and `from catanatron.players.weighted_random import WeightedRandomPlayer`. (`RandomPlayer` *is* `random.choice(playable_actions)`; `WeightedRandomPlayer` just up-weights city > settlement > dev-card.) **Learn:** the golden rule in practice — reuse the library, don't clone it.
2. **[write] `agents/__init__.py` — a name→class registry.** One dict, e.g. `AGENTS = {"random": RandomPlayer, "weighted": WeightedRandomPlayer, "AB": AlphaBetaPlayer}`, so the arena/CLI resolve agents by short name. ~10 lines, zero game logic; you'll append your own classes here later. **Learn:** the registry pattern — decoupling "which agent" (config) from "run a match" (code). *(Housekeeping: every package folder — `arena/`, `env/`, `search/`, … — also needs an `__init__.py`, but those stay **empty**; the file just marks the folder importable. This one is special only because it holds code.)*
3. **[write] `arena/arena.py` — the match runner.** Given a list of agents and `n_games`, for each game build `Game(players, seed=k)`, **shuffle which agent gets which color/seat**, run `game.play()`, and record the winner (`None` = turn-limit, counted separately). Return per-agent wins, win-rate, mean VP (`get_actual_victory_points`), mean turns. Parallelize with `multiprocessing` — a random-bot game is ~0.13 s, so 1,000 games is a couple minutes. **Learn:** controlling for confounds (seat/turn-order bias) and reproducibility via seeds.
4. **[write] `arena/stats.py` — the statistics.** Add **Wilson 95% confidence intervals** on each win-rate, and an **Elo** (or TrueSkill) rating from the match results. **Learn:** why 52% over 100 games is noise but 52% over 5,000 might be real — binomial CIs and rating systems.
5. **[write] `arena/__main__.py` — the CLI entry point.** The filename `__main__.py` is *what makes* `python -m catanbot.arena --agents random,weighted --games 1000` work — `python -m <package>` runs that package's `__main__.py`. It imports `AGENTS`, resolves the `--agents` names to classes, and calls `play_games`. (Optionally `register_cli_player("CB", YourAgent)` to also run inside stock `catanatron-play`.) **Learn:** how `python -m` resolves a package, and keeping the CLI a thin shell over importable logic.
6. **[write] `scripts/gate.py` — a regression gate.** Exit non-zero unless a candidate beats a reference by a threshold with **non-overlapping CIs**. **Learn:** turning "it feels better" into an automated, statistically honest check.

**Watch out:** always run ≥1,000 games and read the CIs before believing an improvement. Catan is high-variance.

**Done when:** the arena prints a ranked table with CIs, and `weighted` beats `random` with non-overlapping intervals.
🎥 The first leaderboard table in your terminal.

---

### Phase 2 — A strong classical bot (your L1)  💻+🖥️

**What you're building:** a bot that plays *well* with hand-written knowledge — a scoring function + shallow look-ahead. **No ML yet.**
**Why / learn:** this becomes three things at once — the **bar** RL must beat, the **opponent** it trains against, and the **teacher** it imitates. **Concept:** evaluation functions and adversarial search (minimax/alpha-beta with chance nodes), plus black-box optimization for tuning.

**Build it — in order:**
1. **[import] Benchmark the ladder first.** In your arena, rank the built-ins weakest→strongest: `W` → `VP` → `F` (`ValueFunctionPlayer` — a 1-ply greedy on the hand-crafted eval, near-instant and surprisingly strong) → `AB:2` (the bar). Note: `AlphaBetaPlayer` has a **20 s/decision cap** — use `prunning=True` + fewer games for AB matchups, and `F` for bulk experiments. **Learn:** how much raw strength comes from a good eval alone (F) vs. from search on top (AB).
2. **[reuse the search — you write only the eval] Understand that you never write alpha-beta.** Catanatron's already handles chance nodes (`expand_spectrum`) and pruning (`minimax.py:84`). You inject your evaluation two ways: **(a) weights only** → `AlphaBetaPlayer(color, params=my_weights)` (no subclass); **(b) new features** → subclass `AlphaBetaPlayer`, set `self.use_value_function = True`, and override `value_function(self, game, p0_color)`; the search calls *your* scorer at every leaf (`minimax.py:97`). **Learn:** how an eval function plugs into a search tree — the Stockfish pattern.
3. **[write] Stage 1 — tune the shipped 13-weight eval.** Start from Catanatron's `DEFAULT_WEIGHTS` and its tuned `CONTENDER_WEIGHTS` (13 knobs: production, VP, longest-road, army, hand synergy, … — Appendix A). Optimize the vector with **CMA-ES in log-space** (weights span `1`→`3e14`), fitness = arena win-rate vs. `F`/`AB:2`. This is embarrassingly parallel → **🖥️ SLURM array jobs** (your first ELSA use). **Learn:** derivative-free optimization (CMA-ES) and running parallel sweeps on a cluster.
4. **[write] Stage 2 — `agents/heuristic.py`: add features the shipped eval lacks.** Reuse `features.py` extractors, don't rewrite them. Candidates: **multi-opponent awareness** (the default "only considers 1 enemy" — score vs. the *leader* in 4p), **robber threat**, **port synergy** (`port_distance_features` exists but is unused), **army/road *contest*** (margin vs. best opponent), **distance-to-next-VP**. **Learn:** feature engineering — turning game intuition into numbers, then measuring whether each feature actually helps.
5. **[write] `env/belief.py` — a hidden-info tracker.** Track each opponent's min/max possible resources from public events (visible via `GameAccumulator.step`). `player_deck_random_select` (`state_functions.py:322`) is how the engine samples a stolen card — mirror it for the determinization you'll need in search. **Learn:** reasoning about hidden information — belief state from public observations.

**Watch out:** a bad belief tracker makes search *worse* than none. Validate it against full-information games first.

**Done when:** your bot beats Catanatron's `AlphaBetaPlayer` at depth 2 (`AB:2`) **>55%** over 1,000 four-player games (non-overlapping CI). *(This bot alone may already win most colonist.io games — it's your fallback if RL stalls.)*
🎥 Your bot overtaking Catanatron's best on the leaderboard.

---

### Phase 3 — First learning agent (RL + imitation)  💻+🖥️

**What you're building:** the encoding + a neural net that *learns* to play — first by imitating your Phase-2 bot, then improving with PPO.
**Why / learn:** the leap from hand-written rules to learning from experience. **Concept:** observation encoding, action masking, policy/value networks, behavioral cloning, and PPO.

**Build it — in order:**
1. **[write] Step 0 — get a learning curve THIS week, with zero custom code.** The stock env is ready: `env = gymnasium.make("catanatron/Catanatron-v0")` gives a 614-float obs, `Discrete(332)` actions, and `env.action_masks()` that plugs straight into **sb3-contrib `MaskablePPO`**. `pip install sb3-contrib torch`, wrap with `ActionMasker`, train vs. the default `RandomPlayer`. Target: **>90% vs. Random**. **Learn:** the full RL loop end-to-end (obs → masked policy → action → reward) before you complicate anything.
2. **[write] Raise the difficulty via config, not code.** `CatanatronEnv(config={"enemies": [AlphaBetaPlayer(Color.RED, prunning=True)], "representation": "mixed"})` — the env auto-plays the enemy between your moves. Curriculum: Random → `W` → `F` → `AB`. `map_type="MINI"` = a cheap, fast board for debugging. **Learn:** curriculum learning, and how the env abstracts opponents away.
3. **[write] `env/encoding.py` — your observation (only after Step 0 works).** Start with `representation="mixed"` (Catanatron already builds a `(C, 21, 11)` board tensor + numeric vector via `gym/board_tensor_features.py`) before hand-rolling anything fancier. **Write down the exact tensor spec** — the live bridge (Phase 6) must reproduce it byte-for-byte. **Learn:** representation design — what a net needs to see, and spatial (CNN) vs. flat features.
4. **[write] `nn/policy.py` — a policy/value net.** Start with the flat `Discrete(332)` head. Move to a **multi-head** design (action-type head over ≈12 of the 18 `ActionType`s once trading is cut, plus node/edge/tile/resource heads) **only when** the flat head plateaus. Always apply the mask to logits (illegal → −∞ before softmax). **Learn:** structured action spaces and masked policies — measure before adding complexity.
5. **[write] `train/bc.py` — behavioral-cloning warm-start.** Generate a big dataset from `F`/`AB` self-play (add `epsilon≈0.05–0.1` for variety) using Catanatron's **`ParquetDataAccumulator`** — it already logs features, board tensors, actions, and returns (**🖥️ ELSA CPU array**). Train the net to predict the teacher's action (cross-entropy) + the game outcome (MSE). **Learn:** supervised pre-training / imitation as an RL warm start (the AlphaGo recipe).
6. **[write] `train/ppo.py` — real PPO.** Graduate from sb3 to **CleanRL**'s single-file PPO once you need the multi-head net or BC-initialized weights. Log to W&B: reward, win-rate vs. each baseline (an arena eval callback every K updates), policy entropy, and the action-type histogram. **Learn:** PPO internals (advantages, clipping, entropy) by reading and modifying a clean implementation.
7. **[write] Checkpoint discipline (🖥️).** Save every N updates and support resume — SLURM walltime *will* kill long jobs. **Learn:** fault-tolerant long-running training.

**Watch out — the classic failure:** **END_TURN collapse**, where the agent learns to spam "end turn." Watch the action histogram; lean on the BC init + sparse (+1/−1) reward; verify masking is correct *before* blaming the algorithm.

**Done when:** the warm-started agent beats Random >95%, WeightedRandom >80%, and is competitive with L1.
🎥 The training curve climbing after the BC seed.

---

### Phase 4 — AlphaZero self-play (the headline)  🖥️

**What you're building:** MCTS guided by your net, in a self-play loop that improves past L1.
**Why / learn:** the marquee technique — a system that teaches itself by playing itself. **Concept:** MCTS (PUCT), self-play data generation, the actor–learner pattern, and handling chance + hidden info in search.

**Build it — in order:**
1. **[write] `search/mcts.py` — PUCT MCTS** using the net for priors + leaf value. Read two in-repo references first: `MCTSPlayer` (`players/mcts.py:16`, plain UCT — your structural template) and `expand_spectrum` (chance-node outcomes with probabilities — reuse for dice). `game.copy()` + `game.execute()` are your rollout primitives. Handle: **dice** (chance nodes), **hidden cards** (determinize per simulation from `env/belief.py` — i.e. Information-Set MCTS), and **4 players** (max-n; the net outputs a per-player value vector). **Learn:** search under stochasticity and imperfect information.
2. **[write] `train/selfplay.py` — the AlphaZero loop.** **Actors** (🖥️ CPU array) play games with MCTS-guided net and store `(state, MCTS visit-count policy π, outcome z)`; the **learner** (🖥️ GPU) trains the net to match π and z; a **gate** promotes a new net only if it beats the current best in the arena; mix in **past generations + L1** as opponents to avoid collapse. **Learn:** the self-improvement loop and why the gate + opponent diversity keep it stable.
3. **[write] Distributed plumbing (§7).** Actors pull latest weights / push games; learner pushes weights. Start with a **shared filesystem** on ELSA scratch; upgrade to Redis/gRPC only if I/O-bound. **Learn:** the actor–learner architecture behind scaled RL.
4. **[write] Validate small, then scale.** Run few MCTS sims (25–50) for fast iterations and confirm **Elo rises across generations** *before* spending big compute. **Learn:** cheap-first experimentation — prove the loop learns before you pay for scale.
5. **[write] Add trading back — last.** Only once the no-trade agent beats L1; extend the action heads for variable-length offers and expect a sample-efficiency hit. **Learn:** curriculum/scope control in a hard action space.

**Watch out:** this is open-ended **research**, not a checklist. Budget weeks, checkpoint everything, and treat any consistent Elo gain as a win. Determinized MCTS is subtle — unit-test it on tiny positions.

**Done when:** an AlphaZero agent beats your L1 bot head-to-head and keeps improving across generations (rising Elo).
🎥 The generational Elo ladder climbing; the ELSA job dashboards ("render farm for Catan").

---

### Phase 5 — The colonist.io live interface  💻 (never on ELSA)

**What you're building:** code that *sees and clicks* a real colonist.io game — driven by a dumb scripted policy, **no AI yet**.
**Why / learn:** prove the "eyes and hands" work before attaching the brain. **Concept:** reverse-engineering a network protocol, and browser automation of a canvas/WebGL app. Full detail in §6.

**Build it — in order:**
1. **[write] `live/sniffer.py` — capture the protocol.** Watch WS frames in Chrome DevTools by hand; then a **Tampermonkey** userscript hooking `WebSocket.prototype` forwards frames to a local Python server. Log many full games. **Learn:** observing and logging a live protocol.
2. **[write] `live/protocol.py` — parse to your internal state.** Catalog message types (board layout, dice, builds, robber, card gains/losses, dev cards, turn/phase) and map them onto **the same state your agents consume**. **Learn:** writing a parser/state-machine from observed data.
3. **[write] State mirror + reconciliation.** Rebuild live game state from the stream; periodically check it against the page; detect desync. **Learn:** keeping a shadow model in sync with an external source of truth.
4. **[write] `live/executor.py` — act by clicking.** The board is WebGL (no DOM nodes for spots), so build a **board-entity → pixel-coordinate map** and drive real clicks via **Playwright/CDP** with human-like jitter. Cover roll/build/robber/discard/dev-card sub-flows. **Learn:** GUI automation and coordinate mapping.
5. **[write] `live/bridge.py` (skeleton).** Wire sniffer → state → a **scripted** policy → executor. The real agent comes in Phase 6. **Learn:** integration seams — build the pipe before the payload.

**Watch out:** ToS/AUP — **throwaway account**, run on your **own PC**, log in manually (don't automate hCaptcha), pace like a human.

**Done when:** with a scripted policy, the bridge completes a full colonist.io game vs. the site's bots / an easy lobby.
🎥 The first time your code clicks through a real game by itself.

---

### Phase 6 — Bridge the brain + go live  💻

**What you're building:** your trained agent playing real online games, then improving from real data.
**Why / learn:** the payoff integration — and a lesson in sim-to-real gaps. **Concept:** deployment, latency budgets, and closing the data loop.

**Build it — in order:**
1. **[write] The adapter — reuse the encoding unchanged.** Convert live state → agent observation and agent action → executor click by calling `env/encoding.py` (the single source of truth). This is *why* the offline encoding had to be exact. **Learn:** why a shared representation makes sim-to-real trivial.
2. **[write] Latency budget.** Make `decide()` (incl. MCTS) fit colonist.io's turn timer; tune MCTS sims to the time available. **Learn:** anytime algorithms — trading think-time for strength.
3. **[write] Edge cases + safe fallback.** Handle robber/discard/dev-card/trade flows; on parser desync, fall back safely (end turn / graceful resign). **Learn:** defensive engineering against a messy real world.
4. **[write] Supervised → unsupervised runs.** Watch first, ready to intervene; then let it run on the throwaway account and collect logs. **Learn:** staged rollout.
5. **[write] The data flywheel.** Feed real game logs back as evaluation (and optional fine-tuning) — humans play differently from self-play. **Learn:** closing the loop between deployment and training.

**Watch out:** real opponents + UI timing surface bugs your sim never did — budget a debugging tail.

**Done when:** the bot plays full live games unattended and wins a meaningful fraction.
🎥 The bot winning a live game in real time (the money shot).

---

### Phase 7 — YouTube production  💻

**What you're building:** the video that makes it all worth it.
**Why / learn:** communicating a technical project to a general audience — a skill in itself.

**Build it:**
1. **Capture from day one:** web-UI games, arena/Elo curves, terminal logs, the first time it beats you, ELSA dashboards, live games.
2. **Narrative arc:** the goal → the naive approaches that fail (great hook) → baselines → teaching by imitation → self-play on a 99-GPU cluster → watching it climb → live games.
3. **Honest framing (§8):** "I built a Catan AI / beat the strongest open-source bot / tested it vs. the internet" — not "cheated to #1." Be transparent about botting.
4. **Assets:** a clean version of the §4 diagram, training-curve animations, a highlight reel.

**Done when:** a published video (and, optionally, a public repo).

---

## 6. The colonist.io interface (deep dive)

The one part with no turnkey solution — **probe it early (around Phase 2)** so surprises surface before you depend on them.

**6a. Reading state (do this first — passive, low-risk).** colonist.io sends game state over **WebSockets as plain JSON** with descriptive fields. Capture, least→most robust: DevTools by hand → a Tampermonkey userscript forwarding frames to a local server → mitmproxy/CDP for the production bridge. Parse into the *same* internal state your agents consume.

**6b. Acting (two paths).**
- **Path A — browser automation (recommended):** drive Chrome via Playwright/CDP; the board is a **canvas/WebGL** element, so map board entities → pixel coordinates and click with human-like jitter. Robust to protocol changes; you maintain the coordinate map.
- **Path B — WebSocket injection:** craft outgoing action messages directly. Fast but brittle, and you must reverse any sequence/auth fields — more clearly automated, thus more detectable.

Recommendation: **read via WebSockets, act via browser automation.**

**6c. Real-world robustness.** Human-like timing/jitter; log in manually (avoid hCaptcha); handle reconnects, turn timers, and unexpected modals; add a **state-reconciliation** check that bails/alerts on desync.

---

## 7. Compute & ELSA

Map each machine to its best role; don't build the distributed setup until single-machine self-play already learns.

| Hardware | Role |
|---|---|
| **TCNJ ELSA** (~2,916 CPU cores, 99 GPUs, 25 TB RAM) | the engine — parallel self-play (CPU), training (GPU), sweeps. **Not** the live bot (AUP + needs a browser). |
| **Gaming PC (best GPU)** | primary dev + learner; **host for the live colonist.io bridge**. |
| **Laptop (GPU)** | dev & evaluation — arena runs, watching games, small experiments. |
| **Old desktops** | optional extra CPU actors; largely redundant once you have ELSA. |

**Using ELSA (the practical bits):**
- **Jobs go through SLURM** (`sbatch`, `srun`, `squeue`), default partition `short`; never run training on the login node.
- **No root needed:** `module load miniconda3` (or install your own Miniconda in `$HOME`), then `conda create -n catan python=3.11 && pip install ...`. For full reproducibility, use an **Apptainer/Singularity** container.
- **GPU jobs:** `--gres=gpu:1` + `module add cuda`; match your PyTorch build to ELSA's CUDA.
- **Compute nodes often have no internet** → `pip install` on the login node into your `$HOME` env.
- **Walltime + queue** → checkpoint frequently and resume; use **array jobs** for sweeps and parallel self-play.
- **Storage:** self-play data is large → use scratch/project space, not your home quota.
- **AUP:** keep it to legit ML training; run the live bot on your own machine. If unsure, ask HPC staff or work under a faculty umbrella.

**Distributed self-play (build only after it learns on one machine):**
```
 [ELSA CPU array: actors] --games--> [shared scratch / replay buffer] --> [ELSA GPU: learner]
        ▲ pull latest weights                                                    │ push weights
        └────────────────────────────────────────────────────────────────────────┘
```
**Honest note:** more actors speed up *data generation*, not *learning per sample*. ELSA lets you run the settlers-rl-scale experiment far faster than one 3090 — but Catan RL is still hard and open-ended, so budget weeks and checkpoint everything.

---

## 8. Risks, ethics & ToS

- **colonist.io almost certainly forbids bots.** Expect a detected account to be **permanently banned** and ladder placements wiped → **throwaway account only, never your main.**
- **"Botted to #1" can backfire** as a video — many see automating vs. humans as cheating, and the result is unverifiable/erasable.
- **Lower-risk flexes that make an equally good (often better) video:** (1) **beat Catanatron's strongest bot** / win a bot-vs-bot tournament (fully reproducible); (2) beat strong humans in a controlled/exhibition setting; (3) frame the live portion as "I tested my AI against the internet," transparently.
- **If you go on the ladder:** human pacing, don't grind 24/7, be transparent in the video, and accept a possible ban.

This is a legitimate "I built an AI to play X" project — the goal here is honesty and not getting blindsided, not avoidance.

---

## 9. Timeline

Part-time estimates for a strong programmer. **Phase 4 is open-ended research — its dates are aspirational.**

| Phase | Scope | Rough effort |
|---|---|---|
| 0 | Setup, run sims | 2–4 days |
| 1 | Baselines + arena | ~1 week |
| 2 | Strong search bot (L1) | 1–2 weeks |
| 3 | RL infra + PPO learning | 2–4 weeks |
| 4 | AlphaZero self-play (L2) | 4–12+ weeks (research) |
| 5 | colonist.io interface (parallel) | 2–4 weeks |
| 6 | Live bridge + laddering | 1–2 weeks |
| 7 | Video production | ongoing capture + ~1–2 weeks edit |

**First demoable win:** end of Phase 2. **First "wow":** Phase 5/6 (your AI playing a real colonist.io game).

---

## 10. Environment setup (done)

**✅ Done & smoke-tested 2026-07-07** — `.venv` on Python 3.11, Catanatron 3.3.0 editable-installed from the local clone. Kept here so it's reproducible on a fresh machine.

> **SSL note (TCNJ proxy):** the network does TLS interception, so pip must trust the Windows cert store. Recent pip (ours is 26) enables `truststore` automatically; older pip needs the one-time bootstrap in step 2.
> **Moved-repo note:** a moved/renamed repo **breaks the venv** (exe shims + editable `.pth` bake in absolute paths) — recreate `.venv` and reinstall after any move.

```powershell
cd C:\Users\Tyler\PycharmProjects\Clankerless

# 1. Virtual environment on Python 3.11
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. (older pip only) make pip trust the Windows cert store
python -m pip install --upgrade --trusted-host pypi.org --trusted-host files.pythonhosted.org pip truststore

# 3. Editable-install Catanatron (cloned at .\catanatron) with RL + dev extras
pip install -e ".\catanatron[gym,dev]"

# 4. Smoke test
catanatron-play --players=R,R,R,R --num=10
```

**PyCharm:** point the interpreter at `.venv\Scripts\python.exe` via *Add Interpreter → **Select existing***.

---

## Appendix A — Catanatron API map (verified)

Read from the clone; **src root** = `catanatron/catanatron/catanatron/`. Sizes marked *(measured)* came from running the package in `.venv`.

### Core loop
| Symbol | Where | Notes |
|---|---|---|
| `Player.decide(self, game, playable_actions) -> Action` | `models/player.py:37` | The interface every brain implements. `game` is read-only. `reset_state()` between games (`:47`). |
| `Action = namedtuple(color, action_type, value)` | `models/enums.py:113` | Immutable; `value` shape is per-type (`enums.py:75–106`). |
| `ActionType` (**18** members) | `models/enums.py:68` | `ROLL, MOVE_ROBBER, DISCARD_RESOURCE, BUILD_ROAD/SETTLEMENT/CITY, BUY_DEVELOPMENT_CARD, PLAY_KNIGHT/YEAR_OF_PLENTY/MONOPOLY/ROAD_BUILDING, MARITIME_TRADE, OFFER/ACCEPT/REJECT/CONFIRM/CANCEL_TRADE, END_TURN`. |
| `ActionPrompt` | `models/enums.py:58` | Which sub-decision is faced now: `BUILD_INITIAL_*, PLAY_TURN, DISCARD, MOVE_ROBBER, DECIDE_TRADE, DECIDE_ACCEPTEES`. |
| `Game(players, seed=None, discard_limit=7, vps_to_win=10, ...)` | `game.py:95` | `seed` ⇒ reproducible games. |
| `game.play(accumulators=[]) -> Color \| None` | `game.py:132` | `None` = hit `TURNS_LIMIT`. `play_tick()` = one ply. `game.copy()` + `game.execute(action)` = search primitives. |
| `game.playable_actions` | `game.py:130` | Legal actions now (from `generate_playable_actions`, `models/actions.py:46`). |
| `GameAccumulator.before / step / after` | `game.py:51` | Hook every action of every game — stats, logging, data capture. |

### Built-in players — `cli/cli_players.py:16`
| Code | Class @ file | Role |
|---|---|---|
| `R` | `RandomPlayer` `models/player.py:84` | floor |
| `W` | `WeightedRandomPlayer` `players/weighted_random.py:14` | weak (city 10000 > settlement 1000 > dev 100) |
| `VP` | `VictoryPointPlayer` `players/search.py:10` | greedy-VP |
| `F` | `ValueFunctionPlayer` `players/value.py:144` | 1-ply greedy on the hand-crafted eval — fast + strong |
| `AB:d:prune` | `AlphaBetaPlayer` `players/minimax.py:18` | **the bar** — `depth=2` default, chance-node expectation, 20 s/decision cap (`minimax.py:15`) |
| `SAB` / `M:n` / `G:n` | `SameTurnAlphaBetaPlayer` / `MCTSPlayer` (`mcts.py:16`) / `GreedyPlayoutsPlayer` | AB within-turn / UCT reference / N-playout greedy |

`register_cli_player("CB", YourAgent)` (`cli/cli_players.py:86`) runs your agent inside `catanatron-play`.

### Evaluation function (Phase 2) — `players/value.py`
- `DEFAULT_WEIGHTS` (13 keys, `value.py:20`): `public_vps, production, enemy_production, num_tiles, reachable_production_0/1, buildable_nodes, longest_road, hand_synergy, hand_resources, discard_penalty, hand_devs, army_size`. Range `1`→`3e14` ⇒ tune in log-space.
- `CONTENDER_WEIGHTS` (`value.py:40`) = author's tuned set — the number to beat.
- Inject via `AlphaBetaPlayer(color, params=<dict>)`, or subclass + override `value_function` (`use_value_function=True`; called at each leaf, `minimax.py:97`). Known gap: "only considers 1 enemy player" (`value.py:148`).
- Extractors in `features.py`: `create_sample`, `build_production_features`, `reachability_features`, `expansion_features`, `port_distance_features`, `graph_features`.

### Gym env (Phase 3) — `gym/envs/catanatron_env.py`
- `import catanatron.gym; gymnasium.make("catanatron/Catanatron-v0")` *(verified)*.
- **Spaces (measured):** obs **614** floats (2p); `Discrete(332)` (2p BASE) / `Discrete(370)` (4p). Feature counts 2p/3p/4p = 614/808/1002.
- `config` (`:51`): `enemies=[Player,…]`, `reward_function` (default `simple_reward` +1/0/−1, `:33`), `map_type BASE/TOURNAMENT/MINI`, `vps_to_win`, `representation "vector"|"mixed"`.
- `representation="mixed"` ⇒ `{board:(C,21,11), numeric:vector}` (`:81`).
- **`env.action_masks()` (`:120`) is sb3-contrib `MaskablePPO`-ready.** Env auto-plays enemies (`_advance_until_p0_decision`, `:204`); truncates at `TURNS_LIMIT` or 10 invalid actions.
- Data capture: `gym/accumulators.py` — `ParquetDataAccumulator` (`:181`) logs features, board tensor, action, discounted return.

---

## Appendix B — Catanatron state & data model (verified)

Mirror this in `env/encoding.py`, `env/belief.py`, and the live bridge. Sources: `state.py`, `state_functions.py`, `models/decks.py`, `models/enums.py`.

**`game.state: State` (`state.py:46`):**
- **`player_state: dict`** — flat, keys `P{i}_{FIELD}` (`PLAYER_INITIAL_STATE`, `state.py:22`): `VICTORY_POINTS, ACTUAL_VICTORY_POINTS, ROADS_AVAILABLE(15), SETTLEMENTS_AVAILABLE(5), CITIES_AVAILABLE(4), HAS_ROAD/ARMY/ROLLED, LONGEST_ROAD_LENGTH, {RES}_IN_HAND, {DEV}_IN_HAND, PLAYED_{DEV}, …`. `VICTORY_POINTS` is public; `ACTUAL_VICTORY_POINTS` adds hidden VP cards.
- **`buildings_by_color[color][SETTLEMENT|CITY|ROAD]`** → node/edge ids (`state.py:66`).
- **`resource_freqdeck`** = bank `[WOOD,BRICK,SHEEP,WHEAT,ORE]`, start `[19]*5` (`decks.py:32`); `development_listdeck` = shuffled dev bank.
- **`board`** (`models/board.py:39`): `board.map`, `board.buildable_node_ids(color)`, `board.map.adjacent_tiles[node_id]`, `board.map.land_tiles`.
- **Turn/phase flags:** `is_initial_build_phase, is_moving_knight, is_discarding, is_road_building, is_resolving_trade, current_trade (11-tuple)`. `current_player_index` ≠ `current_turn_index` for out-of-turn discards.
- `state.copy()` (`:153`) = fast copy for search.

**Gotchas:**
- **Resources & dev cards are *strings*, not enums** (`enums.py:5–35`): `"WOOD"`, `"KNIGHT"`, `SETTLEMENT="SETTLEMENT"`. Import the `Final` constants; don't invent enum members.
- **freqdeck** = 5-vector histogram. Costs (`decks.py:25`): ROAD `[1,1,0,0,0]`, SETTLEMENT `[1,1,1,1,0]`, CITY `[0,0,0,2,3]`, DEV `[0,0,1,1,1]`.
- **Use accessors** (`state_functions.py`): `player_key`, `get_actual_victory_points`, `get_longest_road_length`, `get_largest_army`, `get_player_freqdeck`, `player_num_resource_cards`, `player_deck_random_select` (the robber-steal draw — mirror for determinization).

**"Cut trading" — there is no config flag.** `generate_playable_actions` (`actions.py:46`) during `PLAY_TURN` emits build/buy/dev/`END_TURN` + **only maritime trades** (`:88`). Domestic `OFFER_TRADE` is *never* auto-generated — it appears only if an agent returns one (`:92,102`). So stock games are already domestic-trade-free; to keep your agent trade-free, just don't emit `OFFER_TRADE` (and optionally mask `MARITIME_TRADE`).

---

## Appendix C — Repo structure

```
Clankerless/
├── plan.md
├── README.md
├── requirements.txt
├── catanatron/                 # the cloned sim (editable-installed; gitignored)
├── pyproject.toml              # YOUR package's config → `pip install -e .` makes `catanbot` importable
├── src/
│   └── catanbot/               # YOUR package — EVERY subfolder below needs its own __init__.py
│       ├── __init__.py         # (empty) marks `catanbot` as a package
│       ├── smoke.py            # Phase 0
│       ├── agents/
│       │   ├── __init__.py     #   the AGENTS registry (real code) — the one non-empty __init__
│       │   ├── heuristic.py    #   Phase 2: your value function
│       │   ├── alphabeta.py    #   Phase 2: AlphaBetaPlayer subclass w/ your eval
│       │   ├── rl_agent.py     #   Phase 3+
│       │   └── base.py         #   add ONLY at Phase 6 (live bridge)
│       ├── arena/
│       │   ├── __init__.py     #   (empty)
│       │   ├── arena.py        #   play_games(): run matches → win table
│       │   ├── stats.py        #   Wilson CIs + Elo
│       │   └── __main__.py     #   the CLI → enables `python -m catanbot.arena`
│       ├── env/
│       │   ├── __init__.py     #   (empty)
│       │   ├── encoding.py     #   state<->tensor, action<->id, masks
│       │   ├── belief.py       #   opponent hidden-info tracker
│       │   └── selfplay_env.py
│       ├── search/             # __init__.py + mcts.py, determinization, tree utils
│       ├── nn/                 # __init__.py + policy.py (policy/value net)
│       ├── train/              # __init__.py + bc.py, ppo.py, selfplay.py
│       └── live/               # __init__.py + sniffer, protocol, executor, bridge
├── scripts/                    # gate.py, SLURM job scripts
├── notebooks/                  # analysis & plots
├── data/                       # self-play games, checkpoints (gitignored)
└── tests/
```

*(Note: the package is named `catanbot` while the repo is `Clankerless` — harmless, but say the word and I'll rename the package to match.)*

---

## Appendix D — Resources

**Framework:** Catanatron [repo](https://github.com/bcollazo/catanatron) · [docs](https://docs.catanatron.com) · [Discord](https://discord.gg/FgFmb75TWd)

**Must-read prior art:**
- "5 Ways NOT to Build a Catan AI" (Catanatron author): https://medium.com/@bcollazo2010/5-ways-not-to-build-a-catan-ai-e01bc491af17
- "Learning To Play Settlers of Catan With Deep RL" (settlers-rl): https://settlers-rl.github.io/
- "Modeling Catan through self-play" (Justin Asher): https://justinasher.me/catan_ai
- QSettlers — Deep RL for Catan: https://akrishna77.github.io/QSettlers/

**colonist.io:** "Abusing my CS knowledge to cheat at Catan" (WS reverse-engineering): https://medium.com/@alberttheblacksheep/abusing-my-computer-science-knowledge-to-cheat-at-catan-a0f72fa30309 · [colonist.io](https://colonist.io/)

**RL background:** AlphaZero paper: https://www.science.org/doi/10.1126/science.aar6404 · CleanRL (single-file PPO): https://github.com/vwxyzjn/cleanrl · sb3-contrib MaskablePPO: https://sb3-contrib.readthedocs.io/
