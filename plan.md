# CatanBot — Project Plan

> Goal: build an AI that plays online Catan on **colonist.io** and wins consistently, with the stretch goal of climbing the ranked ladder and producing a YouTube video about the build.
>
> Chosen approach: **Deep RL self-play (AlphaZero/PPO) as the north star**, built on top of a fast simulator and strong search baselines.
>
> Last updated: 2026-06-16 (v2 — detailed per-step build guide + ELSA HPC integrated)

---

## Table of contents
1. [North star & success criteria](#1-north-star--success-criteria)
2. [Reality check (read this first)](#2-reality-check-read-this-first)
3. [Strategy: why this order](#3-strategy-why-this-order)
4. [Tech stack & key decisions](#4-tech-stack--key-decisions)
5. [System architecture](#5-system-architecture)
6. [Phased roadmap](#6-phased-roadmap)
7. [The colonist.io live interface (the hard, novel part)](#7-the-colonistio-live-interface-the-hard-novel-part)
8. [Compute & distributed self-play plan](#8-compute--distributed-self-play-plan)
9. [Risks, ethics & Terms-of-Service](#9-risks-ethics--terms-of-service)
10. [Rough timeline & milestones](#10-rough-timeline--milestones)
11. [Immediate next steps (first coding session)](#11-immediate-next-steps-first-coding-session)
12. [Suggested repo structure](#12-suggested-repo-structure)
13. [Resources](#13-resources)

---

## 1. North star & success criteria

The project is "done" in stages. Define success at each level so you always have a win to show:

| Level | Success criterion | Why it matters |
|------|-------------------|----------------|
| **L0 — Plumbing** | Run thousands of simulated games/sec; an evaluation arena reports win-rates & Elo | You can't improve what you can't measure |
| **L1 — Strong classical bot** | Beat Catanatron's best built-in bot (AlphaBeta) > 55% in 4-player games | A genuinely strong, shippable agent + a hard sparring partner for RL |
| **L2 — Learned agent** | An RL agent that beats the L1 bot head-to-head | Proof the self-play loop actually learns |
| **L3 — Live integration** | Bot reads colonist.io game state and plays full games end-to-end | The unique, video-worthy engineering feat |
| **L4 — Ladder** | Positive win-rate vs. real ranked opponents; climbing | The headline goal (see ethics/ToS section) |

**Pick L1 + L3 as the minimum bar for a great video.** L4 is the dream but is the riskiest and least in your control (see §9).

---

## 2. Reality check (read this first)

Two well-documented prior attempts set expectations honestly:

- **Catanatron author (bcollazo)** tried DQN, cross-entropy RL, supervised learning, TensorForce, and pure MCTS. **All lost to a simple hand-crafted evaluation function + depth-2 Alpha-Beta search** (Stockfish-style). Pure MCTS was too slow (~3s/turn for 100 sims); RL on random-play data was too noisy; the action space ballooned to 5,000+ and had to be compressed to ~289.
- **settlers-rl** (deep RL, PPO) trained ~**1 month on an RTX 3090 + 32-core / 128GB**, ~**450M decisions**, and the result was *"still not close to the standard of a good human player."* Adding MCTS-style search **on top of** the learned policy improved it. Their explicit advice: **drop trading at first** — it's the hardest part and kills sample efficiency.

**Takeaways that shape this plan:**
1. The strongest *known* Catan AI is **search + a good evaluation function**, not raw RL. So we build that first — it's both a strong shippable bot *and* the benchmark/sparring partner RL needs.
2. The realistic high-ceiling target is **AlphaZero-style: a neural net (policy + value) guided by MCTS**, trained via self-play. This is the marriage of the two ideas, and the research consistently shows **search on top of a policy** is what works.
3. **Simulator speed is everything** for self-play. Millions of games must be cheap. This is why we use Catanatron (thousands of games/sec), not a slower academic framework.
4. **Cut trading initially.** Catan without player-to-player trading is still a complete, winnable game and is *much* more tractable to learn. Add it back as an advanced phase.

This isn't a downgrade of your RL goal — it's the professional path *to* it. Every serious AlphaZero project starts with a fast env and strong baselines.

---

## 3. Strategy: why this order

```
        fast sim + arena         strong search bot          RL self-play            live bridge
        (measure everything) ──▶  (the bar to beat)  ──▶   (beat the bar)   ──▶   (colonist.io)
              L0                        L1                      L2                   L3 / L4
```

- You need the **arena (L0)** before anything, or you'll fool yourself about whether changes help.
- The **search bot (L1)** is not throwaway: it's your strongest opponent for self-play bootstrapping, your evaluation baseline forever, and a fully shippable bot if RL underdelivers.
- **RL (L2)** then has a clear, hard target (beat L1) and good opponents to learn against.
- The **live bridge (L3)** is *independent* of which brain you plug in, so it can be built in parallel by a separate work-stream once the agent API is stable. Start sniffing the protocol early — it's the riskiest unknown.

---

## 4. Tech stack & key decisions

| Concern | Decision | Rationale |
|--------|----------|-----------|
| **Simulator / env** | **Catanatron** (`bcollazo/catanatron`) | Fastest Python Catan sim (thousands of games/sec), built-in Gym interface, strong baseline bots, web UI to watch games, active project. **Use this instead of `kvombatkere/Catan-AI`** — that repo is a fine learning reference but is too slow and less complete for RL self-play, where sim speed is the bottleneck. |
| **Language** | Python 3.11+ | Catanatron requires it; the whole ML ecosystem lives here. |
| **DL framework** | **PyTorch** | Standard for custom AlphaZero/PPO; great Windows + CUDA support. |
| **RL: first learning signal** | **PPO** (Stable-Baselines3 or CleanRL) on the Gym env, trading disabled | Robust, well-documented, fast to get a learning curve. CleanRL = single-file, easy to hack the action masking + multi-head policy. |
| **RL: high-ceiling target** | **Custom AlphaZero** (MCTS + policy/value net, self-play) | The known-best pattern for this kind of game. Worth building yourself for the YouTube story and full control. |
| **Search baseline** | Catanatron's **AlphaBeta** bot + your improved eval function | Proven strongest-known; also your benchmark. |
| **Live interface** | **Read** state via WebSocket sniffing; **act** via browser automation (Playwright/CDP) clicking the canvas | Reading is passive and safe; acting via real clicks is more human-like than injecting WS messages (see §7). |
| **Experiment tracking** | Weights & Biases or TensorBoard | Self-play runs are long; you must track win-rate vs. baselines over time. |
| **Package mgmt** | `venv` + `pip` (or `uv` for speed) | Keep it simple; Catanatron is pip-installable. |

### Action-space & state notes (learned from prior work — bake these in early)
- **Action space:** Catanatron already exposes a compact, enumerated action space (~290 actions) with a list of *valid* actions each turn. **Always use action masking** — never let the policy pick illegal moves.
- **Multi-head policy** (from settlers-rl) handles Catan's structured actions well: an action-type head (~13 types) + specialized heads for corners (54), edges/roads (72), tiles (19), dev cards (5), and resource selection. Log-probs sum across heads for composite actions.
- **Hidden information:** track each opponent's **min/max possible resources** from public info (dice gains, robber, trades). During MCTS, **determinize** by sampling opponents' hidden cards from the feasible set.
- **Reward:** start with **sparse win/loss (+1/−1)**. Beware reward shaping on victory points — it can cause the agent to spam `END_TURN` (class imbalance bit bcollazo). If you shape, do it carefully and ablate it.

### Reuse vs. build: what to take off the shelf
Decide this **per layer** — reuse aggressively at the infra/baseline layer, build only the Catan-specific learning that's the actual point. There is **no downloadable model that wins Catan** at a strong-human level; don't plan around finding one.

| Layer | Reuse? | What to use |
|---|---|---|
| RL algorithm (PPO / MCTS code) | **Reuse, always** | CleanRL / Stable-Baselines3 / RLlib — never hand-roll PPO |
| Strong baseline opponent | **Reuse** | Catanatron `AlphaBetaPlayer(n=2)` — the strongest readily-usable Catan AI (a *search* bot, not a neural net); your benchmark + sparring partner |
| Pre-trained Catan *weights* | **Mostly no** | `henrycharlesworth/settlers_of_catan_RL` ships weights + `play.py`, but the author states it's *not at good-human level*. Use as a sparring partner / sanity check, not a ladder-winner. |
| The actual learned agent | **Build** | The novel work and the video's substance |

**The smart shortcut (warm-start, not from-scratch):** don't train from random. **Behavioral-clone** your policy/value net on a few hundred thousand games generated by Catanatron's AlphaBeta bot, *then* run RL/self-play on top. This is the AlphaGo recipe (imitate strong play first, improve second) and it skips the cold-start random-exploration phase that made the from-scratch attempts cost ~a month. Do this in Phase 3 to seed Phase 4.

---

## 5. System architecture

```
                         ┌───────────────────────────────────────────────┐
                         │                  AGENT API                      │
                         │   decide(game_state, valid_actions) -> action   │
                         └───────────────────────────────────────────────┘
                            ▲              ▲              ▲            ▲
              ┌─────────────┘     ┌────────┘      ┌──────┘     ┌──────┘
        ┌───────────┐      ┌────────────┐   ┌───────────┐  ┌──────────────┐
        │ Random /  │      │ Heuristic +│   │ PPO policy│  │ AlphaZero    │
        │ Weighted  │      │ AlphaBeta  │   │ (net)     │  │ (MCTS + net) │
        └───────────┘      └────────────┘   └───────────┘  └──────────────┘
              │                  │                 │              │
              └──────────────────┴────────┬────────┴──────────────┘
                                          ▼
                    ┌──────────────────────────────────────────┐
                    │   ENV LAYER (wraps Catanatron)            │
                    │   • encoding: state<->tensor, action<->id │
                    │   • action masking                        │
                    │   • gym wrapper (1v1 & 4p self-play)      │
                    └──────────────────────────────────────────┘
                          │                          │
            ┌─────────────▼──────────┐    ┌──────────▼───────────────┐
            │  TRAINING PIPELINE      │    │  ARENA / EVALUATION       │
            │  • self-play actors     │    │  • round-robin matches    │
            │  • replay buffer        │    │  • Elo + win-rate vs base │
            │  • learner (GPU)        │    │  • regression gate        │
            └─────────────────────────┘    └───────────────────────────┘

                    ════════ same Agent API plugs into ════════

                    ┌──────────────────────────────────────────┐
                    │   LIVE BRIDGE (colonist.io)               │
                    │   sniffer → protocol parser → internal    │
                    │   state → agent.decide() → action         │
                    │   executor (browser/canvas clicks)        │
                    └──────────────────────────────────────────┘
```

The key design rule: **every brain implements the same `decide()` interface**, and the **same interface is what the live bridge calls**. That means you can develop, train, and evaluate entirely offline, then swap the trained brain into the live game with zero changes to the integration layer.

---

## 6. Phased roadmap

Each phase below lists: **Goal** → **Build** (the concrete tasks / files to create) → **Watch out** (gotchas) → **DoD** (Definition of Done — don't move on until it's met) → 🎥 (what to capture for the video). File paths refer to the structure in §12. Where-it-runs tags: **💻 local** (gaming PC/laptop), **🖥️ ELSA** (cluster), **💻+🖥️** (both).

> Phases 0–4 and 5 can overlap: start the colonist.io protocol work (Phase 5) in parallel as soon as Phase 2 is underway — it's the riskiest unknown and is independent of which brain you build.

---

### Phase 0 — Setup & foundations  💻
**Goal:** a working environment, Catanatron running, and a real understanding of its API.

**Build**
1. **Local env:** Python 3.11 venv; `pip install "catanatron[gym]"`; run `catanatron-play --players=R,R,R,R --num=100` (see §11).
2. **Scaffold the repo** per §12: create `src/catanbot/` package tree, `pyproject.toml` (or `requirements.txt`), and a `.gitignore` (`venv/`, `data/`, `*.pt`, `__pycache__/`, `wandb/`).
3. **Learn the API by reading source:** `Game`, `Player.decide(game, playable_actions) -> Action`, `Color`, the `Action` namedtuple + `ActionType` enum, and `game.state`. Write `src/catanbot/smoke.py` (the snippet in §11).
4. **Inspect the state object:** print `game.state` and walk `catanatron.models` (board, enums, player state) so you know exactly how board/buildings/resources/dev-cards are stored — you'll mirror this representation everywhere.
5. **Watch a game:** stand up the Catanatron web UI (Docker) and/or dump game JSON; visually confirm the rules behave.
6. **Pin versions:** freeze requirements and record the Catanatron version.

**Watch out:** confirm `py -3.11` resolves to 3.11+; Catanatron requires it. Docker is only needed for the web UI, not for the core library.

**DoD:** a script runs 1,000 games in a loop and prints win-rate per color; you can explain the `Player`/`Action` API in your own words.
🎥 Screen-record the very first random-bot game in the web UI.

---

### Phase 1 — Baselines + evaluation arena (the backbone)  💻
**Goal:** the measurement infrastructure. You cannot improve what you can't measure, so this comes before any "smart" bot.

**Build**
1. **`agents/base.py`** — your `Agent` abstraction (a thin subclass of Catanatron's `Player`) exposing `decide(game, playable_actions) -> Action`. Everything (baselines, search, RL, live bridge) implements this one interface.
2. **`agents/random_agent.py`** — uniform random over `playable_actions`.
3. **`agents/weighted_random.py`** — bias toward build city > settlement > road > buy dev > end turn; compare against Catanatron's `WeightedRandomPlayer`.
4. **`arena/arena.py`** — run N games among a list of agents. Must: randomize seating/colors (remove turn-order bias), accept a seed (reproducibility), and report win count, win-rate, mean VP, mean game length. Parallelize with `multiprocessing` (you'll run this constantly).
5. **`arena/stats.py`** — **Wilson 95% confidence intervals** on win-rates, plus an Elo (or TrueSkill) rating from match results. CIs are non-negotiable given dice variance.
6. **`arena/cli.py`** — `python -m catanbot.arena --agents random,weighted --games 1000 --players 4`.
7. **`scripts/gate.py`** — regression gate: exit non-zero if a candidate doesn't beat a reference by a threshold with non-overlapping CIs. Use it before every "this is better" claim.

**Watch out:** a 52% win-rate over 100 games is noise. Run ≥1,000 games and look at CIs. Catan has high variance — small edges need large samples.

**DoD:** the arena prints a ranked table with CIs; `weighted` beats `random` with non-overlapping intervals.
🎥 The first "leaderboard" table printing in the terminal.

---

### Phase 2 — Strong classical bot (your L1 / the bar to beat)  💻+🖥️
**Goal:** a genuinely strong search bot. It's your permanent benchmark, your RL sparring partner, **and** the teacher for the imitation warm-start in Phase 3.

**Build**
1. **Benchmark the ceiling:** wire Catanatron's `AlphaBetaPlayer(n=2)` and `ValueFunctionPlayer` into the arena; record their win-rates as the bar.
2. **`agents/heuristic.py` — evaluation function** scoring a position from one player's POV. Features:
   - **production** = Σ over your settlements/cities of (adjacent tile pip count × resource-scarcity weight), cities count double;
   - current **VP** and distance-to-next-VP;
   - **longest road** length / hold / contest; **largest army** progress (knights played);
   - **dev-card** value (VP cards, knights, expected value of unflipped);
   - **resource diversity** & **port** access (2:1 / 3:1); **expansion potential** (legal build spots left);
   - **robber threat** (your blocked tiles); **defensive term** = minus the max opponent's score.
3. **`search/alphabeta.py`** — depth-2 search. Catan has dice (chance) and hidden info, so use **expectimax over dice outcomes** at chance nodes and **determinize** opponents' hidden cards (sample or expected). Alpha-beta prune; keep depth shallow for speed.
4. **`env/belief.py` — hidden-info tracker:** maintain min/max possible resource counts per opponent from public events (dice gains, robber steals, trades, buys). Feeds determinization here and MCTS later.
5. **Weight tuning (🖥️ ELSA):** optimize eval weights with **CMA-ES** or evolutionary search; fitness = win-rate vs. the AlphaBeta baseline. This is embarrassingly parallel → run candidate evaluations as **SLURM array jobs** (first real ELSA use).
6. **Performance:** cache static board info; profile `decide()` so it'll fit colonist.io's per-turn timer later.

**Watch out:** determinization quality matters — a bad belief tracker makes search worse than no search. Validate the tracker against full-information games first.

**DoD:** your bot beats Catanatron `AlphaBetaPlayer(n=2)` **>55%** over 1,000 four-player games (non-overlapping CI).
> This bot alone may already win most colonist.io games — it's your shippable fallback if RL stalls.
🎥 Your bot overtaking Catanatron's best on the arena leaderboard.

---

### Phase 3 — RL infrastructure + first learning signal (+ imitation warm-start)  💻+🖥️
**Goal:** the encoding/env/network scaffolding, a PPO agent that demonstrably learns, and a behavioral-cloning warm-start so RL doesn't start from random.

**Build**
1. **`env/encoding.py` — observation encoding:** game state → tensors using the 3-module decomposition (tiles ×19, current-player, opponents) from settlers-rl; spatial board features + bucketed numeric features. **Document the exact tensor spec** — the live bridge (Phase 6) must reproduce it byte-for-byte.
2. **`env/encoding.py` — action encoding + mask:** map Catanatron's `playable_actions` ↔ a flat index space and build a boolean **action mask** over the full space. Most RL bugs hide here — unit-test it hard.
3. **`env/selfplay_env.py`** — a Gymnasium env wrapping Catanatron. Start **1v1, trading disabled**, opponent = frozen L1 bot. `step()` returns obs/reward/mask/done; sparse reward **+1/−1** on win/loss.
4. **`nn/policy.py` — multi-head policy/value net (PyTorch):** shared trunk (the 3-module encoder) → action-type head + heads for corners(54)/edges(72)/tiles(19)/dev(5)/resources + a value head. Apply the mask to logits (illegal → −∞ before softmax); sum log-probs across active heads for composite actions.
5. **`train/bc.py` — imitation warm-start (do this first):** generate a large dataset of (state, action, outcome) from L1 AlphaBeta self-play (hundreds of thousands of games — **🖥️ ELSA CPU array job**), then train the net to predict the bot's action (cross-entropy) and the game outcome (value). This is the AlphaGo seed.
6. **`train/ppo.py` — PPO (🖥️ ELSA GPU):** start from **CleanRL**'s single-file PPO; adapt for masking + multi-head (or SB3 + MaskablePPO for a simpler first pass). Initialize from the BC checkpoint. Log to W&B: reward, win-rate vs. each baseline (eval callback running the arena every K updates), policy entropy, and the action distribution.
7. **Checkpoint discipline (🖥️ ELSA):** save every N updates and support resume — SLURM walltime *will* kill long jobs.

**Watch out:** **END_TURN collapse** — if reward shaping or class imbalance makes the agent spam end-turn, watch the action histogram and lean on the BC init + sparse reward. Verify masking is correct before blaming the algorithm.

**DoD:** the (warm-started) agent beats Random >95%, WeightedRandom >80%, and is competitive with L1.
🎥 The training curve where win-rate vs. baselines climbs after the BC seed.

---

### Phase 4 — AlphaZero self-play (the headline / L2)  🖥️
**Goal:** MCTS + the neural net, in a self-play loop that improves past L1, scaled on ELSA.

**Build**
1. **`search/mcts.py` — PUCT MCTS** using the net for priors (policy) and leaf value. Handle Catan's quirks:
   - **chance nodes** for dice (sample/average outcomes);
   - **hidden info** via **determinization** each simulation (sample opponents' cards from `env/belief.py`) — i.e., determinized / Information-Set MCTS;
   - **4 players:** max-n (each node maximizes the acting player); net outputs a per-player value vector.
2. **`train/selfplay.py` — the AlphaZero loop:**
   - **actors (🖥️ ELSA CPU array):** play self-play games with MCTS-guided current net; store (state, MCTS visit-count policy π, final outcome z) to the replay buffer;
   - **learner (🖥️ ELSA GPU):** sample the buffer, train the net to match π (policy loss) and z (value loss);
   - **gating:** a new net becomes "best" only if it beats the current best in the arena by threshold (prevents regressions);
   - **opponent diversity:** mix in past net generations + L1 to avoid self-play collapse.
3. **Distributed plumbing (cross-ref §8):** actors pull latest weights / push games; learner pushes weights. Start with a **shared filesystem** on ELSA scratch (simplest); upgrade to Redis/gRPC only if I/O-bound.
4. **Validate small before scaling:** run with few MCTS sims (25–50) for fast iterations and confirm **Elo rises across generations** before spending big compute on more sims/nodes.
5. **Inference-time search:** keep running MCTS on top of the policy at play time — the research consistently shows it helps.
6. **Add trading back last:** only after the no-trade agent beats L1. Extend the action heads to variable-length offers; expect a sample-efficiency hit (settlers-rl's warning).

**Watch out:** this is open-ended **research**, not a checklist — budget weeks, checkpoint everything, and treat any consistent Elo gain as success. Determinized MCTS is subtle; test it on small positions.

**DoD:** an AlphaZero agent that beats your L1 search bot head-to-head and keeps improving across generations (rising Elo).
🎥 The generational Elo ladder climbing; SLURM job dashboards / the 99-GPU "render farm" angle.

---

### Phase 5 — colonist.io live interface (parallel from ~Phase 2; runs 💻, never on ELSA)
**Goal:** read live game state, execute actions, and complete a full game with a *scripted* policy (the brain comes in Phase 6). Details in §7.

**Build**
1. **`live/sniffer.py` — protocol capture:** DevTools WS frames by hand first; then a **Tampermonkey userscript** hooking `WebSocket.prototype` to forward frames to a local Python server. Log many full games to disk.
2. **`live/protocol.py` — protocol mapping:** catalog message types (initial board: tiles/numbers/ports/harbors; dice; builds; robber; card gains/losses; dev cards; trade offers; turn/phase; bank) and parse them into **the same internal state your agents consume**. Build it up from recorded games.
3. **Live state mirror + reconciliation:** maintain game state from the stream; periodically check it against the rendered page; detect desync.
4. **`live/executor.py` — action execution:** map board entities (corner/edge/tile IDs) → **canvas pixel coordinates** (board is WebGL — no DOM nodes for spots); drive via **Playwright/CDP** with human-like jitter/delays. Cover sub-flows: roll, build, robber placement, discard, dev-card, accept/decline trade.
5. **`live/bridge.py` (skeleton):** wire sniffer → state → a **scripted/heuristic** policy → executor. Plugging in the real agent is Phase 6.
6. **Robustness:** reconnect, turn timers, unexpected modals, animations; **log in manually** (avoid automating hCaptcha), then attach; rate-limit to human pace.

**Watch out:** ToS/AUP — use a **throwaway account**, run on your **own PC, not ELSA**.

**DoD:** with a scripted policy, the bridge completes a full game on colonist.io vs. the site's bots / an easy lobby.
🎥 The first time your code clicks through a real colonist.io game by itself.

---

### Phase 6 — Bridge the brain + go live  💻
**Goal:** plug the trained agent into the live bridge and play full live games, then iterate from real data.

**Build**
1. **Adapter:** live internal state → agent observation, and agent action → executor action — by **reusing `env/encoding.py` unchanged** (single source of truth; this is why the offline spec had to be exact).
2. **Latency budget:** ensure `decide()` (including MCTS) fits colonist.io's turn timer; tune MCTS sims to the time available.
3. **Edge cases end-to-end:** robber/discard/dev-card/trade flows; define a safe fallback (e.g., end turn / graceful resign) on parser desync.
4. **Supervised runs:** you watch, ready to intervene; log every decision + outcome.
5. **Unsupervised runs** on the throwaway account; collect game logs.
6. **Data flywheel:** feed real game logs back as evaluation (and optional fine-tuning / opponent modeling) — humans play differently from self-play.

**Watch out:** real opponents and UI timing surface bugs your sim never did; expect a debugging tail here.

**DoD:** the bot plays full live games unattended and wins a meaningful fraction.
🎥 The bot winning a live game in real time (the money shot).

---

### Phase 7 — YouTube production (capture throughout!)  💻
**Goal:** turn the build journey into the video.

**Build**
1. **Capture from day one:** web-UI games, arena/Elo curves, terminal logs, the first time it beats you, ELSA job dashboards (render-farm B-roll), live colonist.io games.
2. **Narrative arc:** the goal → the naive approaches that fail (great hook, per §2) → building baselines → teaching it by imitation → self-play on a 99-GPU university cluster → watching it climb → live games.
3. **Honest framing (per §9):** lead with "I built a Catan AI / beat the strongest open-source bot / tested it against the internet" rather than "cheated to #1"; be transparent about botting.
4. **Assets:** turn the §5 architecture ASCII into a clean graphic; training-curve animations; highlight reel of clever plays.
5. **Repo/reproducibility:** clean README; consider releasing the AI/training code but **not** the live-botting bridge (ToS).

**DoD:** a published video (and, optionally, a public repo).

---

## 7. The colonist.io live interface (the hard, novel part)

This is the part with no existing turnkey solution and the highest uncertainty — **start probing it early (around Phase 2)** so surprises surface before you depend on them.

### 7.1 Reading game state (do this first — it's passive & low-risk)
colonist.io communicates game state over **WebSockets as plain JSON with descriptive field names** (confirmed by prior reverse-engineering work). Approach:
1. Open a game in Chrome, open **DevTools → Network → WS**, and watch the frames. Log a full game.
2. Catalog message types: board layout (tiles, numbers, ports), dice rolls, builds, robber moves, card transfers, turn/phase changes, trade offers.
3. Write a **protocol parser** that converts these messages into your internal game state (the same representation your agent consumes offline).

Capture options, from least to most robust:
- **DevTools by hand** — for the initial reverse-engineering session.
- **A userscript (Tampermonkey)** that hooks `WebSocket` and forwards frames to a local Python server — great for continuous capture during development.
- **mitmproxy / CDP (Chrome DevTools Protocol) / Playwright** — programmatic, scriptable capture for the production bridge.

### 7.2 Executing actions (two paths)
- **Path A — Browser automation (recommended start):** drive a real Chrome via **Playwright/CDP**. The board renders to a **canvas/WebGL** element, so you can't click DOM nodes for board spots — instead build a **board-state → pixel-coordinate map** and click canvas positions. Pros: looks like a human, robust to internal protocol changes. Cons: you must maintain the coordinate mapping and handle animations/timing.
- **Path B — WebSocket message injection:** craft and send the *outgoing* action messages directly. Pros: fast, precise. Cons: you must reverse the outgoing protocol *and* any sequence numbers/auth/anti-replay; it's more clearly automated and thus more detectable; brittle to protocol updates.

**Recommendation:** **read via WebSockets, act via browser automation (Path A).** It's the most human-like and the most resilient. Keep Path B as a possible optimization.

### 7.3 Real-world robustness
- Human-like **timing/jitter** between actions; don't act in 5ms.
- Handle **captcha/anti-bot** on login (don't automate login if it triggers hCaptcha — log in manually, then attach).
- Reconnect logic, turn timers, unexpected modals, and the discard/robber/dev-card sub-flows.
- A **state reconciliation** check: periodically confirm your internal state matches what the page shows; bail/alert on mismatch.

---

## 8. Compute & distributed self-play plan

You have four tiers of compute. Map each to its best role; don't build the distributed setup until single-machine self-play already learns.

| Hardware | Role | Notes |
|---|---|---|
| **TCNJ ELSA cluster** (≈2,916 CPU cores, 99 GPUs, 25 TB RAM) | **The engine** — massively parallel self-play game generation (CPU), neural-net training (GPU), and hyperparameter sweeps | SLURM scheduler, Lmod modules, conda in userspace (no root). **Do not run the live colonist.io bot here** (AUP + it needs a real browser). |
| **Gaming PC (best GPU)** | Primary **dev + learner** for fast local iteration; **host for the live colonist.io bridge** | Where you debug before scaling out to ELSA. |
| **Laptop (GPU)** | **Dev & evaluation** — arena runs, watching games, small experiments | Keeps the gaming PC free for long runs. |
| **Old desktops** | Optional extra CPU actors / fallback when the ELSA queue is busy | Largely redundant once you have ELSA. |

### Using ELSA (the practical bits)
- **Shell & jobs:** SSH to the login node; everything heavy runs through **SLURM** (`sbatch`, `srun`, `squeue`, `scancel`). Default partition is `short`. **Never** run training on the login node.
- **Environment, no root needed:** `module load miniconda3` (or install your own Miniconda in `$HOME`), then `conda create -n catan python=3.11 && conda activate catan && pip install "catanatron[gym]" torch ...`. The module conda may be old — creating a fresh 3.11 env (or your own Miniconda) fixes it. For full reproducibility, build an **Apptainer/Singularity** container instead.
- **GPU jobs:** request with `--gres=gpu:1` and `module add cuda`; install the **PyTorch build matching ELSA's CUDA** version.
- **No internet on compute nodes (common):** `pip install` on the **login** node into your env (it lives in `$HOME`, visible to compute nodes); pre-stage downloads.
- **Walltime + queue:** jobs are time-limited and may wait → **checkpoint training frequently and resume**; use SLURM **array jobs** for sweeps and parallel self-play.
- **Storage:** self-play data/checkpoints get large → use **scratch/project space**, not your home quota.
- **AUP:** keep it to legit ML training/research; run the colonist.io live-play bot on your own machine. If unsure, clear it with HPC staff or work under a faculty research umbrella.

### Distributed self-play architecture (build only after it learns on one machine)
```
  [ELSA CPU array: self-play actors] --- games --->  [shared scratch / replay buffer]  ---> [ELSA GPU node: learner]
        ▲ pull latest weights                                                                       │ push new weights
        └───────────────────────────────────────────────────────────────────────────────────────────┘
                          (start with a shared filesystem; upgrade to Redis/gRPC only if I/O-bound)
```

**Honest cost/benefit:** more actors speed up *data generation*, not *learning per sample*. ELSA's 99 GPUs + ~2,900 cores let you run the settlers-rl-scale experiment far faster than their single 3090 — but Catan RL is still hard and open-ended, so budget weeks and checkpoint everything. Bonus: "I trained it on a 99-GPU university supercomputer" is great video material.

---

## 9. Risks, ethics & Terms-of-Service

Be clear-eyed about this now so it doesn't sink the project (or your reputation) later.

- **colonist.io's Terms almost certainly prohibit bots/automation.** Expect that a detected bot account can be **permanently banned**, and any ladder placement wiped. **Use a dedicated throwaway account, never your main.**
- **"I botted to #1 on the ranked ladder" can backfire.** A chunk of any audience views automating against real human opponents as cheating, and the result is unverifiable/erasable. The headline goal is real, but plan the framing.
- **Lower-risk flexes that make an equally good (often better) video:**
  1. **Beat Catanatron's strongest bot** / win a bot-vs-bot tournament — fully reproducible and impressive to a technical audience.
  2. **Beat strong humans in a controlled/exhibition setting** (consenting opponents, clearly labeled).
  3. Frame the live portion as **"I tested my AI against the internet"** with transparency, rather than a stealth ladder-climb.
- **If you do go on the ladder:** human-like pacing, don't grind 24/7, be transparent in the video, and accept the account may be banned.
- This is a legitimate, well-trodden "I built an AI to play <game>" engineering project — the goal here is honesty and not getting blindsided, not avoidance.

---

## 10. Rough timeline & milestones

Part-time estimates for a strong programmer. **Phase 4 is open-ended research — treat its dates as aspirational.**

| Phase | Scope | Rough effort |
|------|-------|-------------|
| 0 | Setup, run sims | 2–4 days |
| 1 | Baselines + arena | ~1 week |
| 2 | Strong search bot (L1) | 1–2 weeks |
| 3 | RL infra + PPO learning | 2–4 weeks |
| 4 | AlphaZero self-play (L2) | 4–12+ weeks (research) |
| 5 | colonist.io interface (parallel) | 2–4 weeks |
| 6 | Live bridge + laddering | 1–2 weeks |
| 7 | Video production | ongoing capture + ~1–2 weeks edit |

**First demoable win:** end of Phase 2 (a bot that beats Catanatron's best). **First "wow":** Phase 5/6 (watching your AI play a real colonist.io game).

---

## 11. Immediate next steps (first coding session)

Run these in PowerShell from the project root. (If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.)

> **✅ Done & smoke-tested 2026-06-16** — `.venv` on Python 3.11, Catanatron 3.3.0 editable-installed from the local clone, SSL handled via `truststore`. Kept here so it's reproducible on a fresh machine.

```powershell
cd C:\Users\Tyler\PycharmProjects\CatanBot

# 1. Virtual environment on Python 3.11
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. SSL: behind a TLS-inspecting network (e.g. TCNJ), make pip trust the Windows
#    cert store. pip >= 26 enables truststore automatically; to bootstrap older pip:
python -m pip install --upgrade --trusted-host pypi.org --trusted-host files.pythonhosted.org pip truststore

# 3. Editable-install Catanatron (cloned at .\catanatron) with RL + dev extras
pip install --use-feature=truststore -e ".\catanatron[gym,dev]"

# 4. Smoke test: 10 games of 4 random bots
catanatron-play --players=R,R,R,R --num=10
```

Then, in code (first script, e.g. `src/catanbot/smoke.py`):
```python
from catanatron import Game, RandomPlayer, Color

players = [
    RandomPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]
print(Game(players).play())  # prints the winning color
```

Checklist for the week:
- [ ] venv + Catanatron installed, CLI smoke test passes
- [ ] Read Catanatron docs: `Player` API (the `decide` method), action enum, `Game`
- [ ] Custom `RandomPlayer` subclass running in a loop over 1,000 games
- [ ] Skeleton **arena** that prints a win-rate table
- [ ] Stand up the Catanatron web UI (Docker) and watch one game
- [ ] (Parallel, low effort) Open colonist.io, watch the WebSocket frames in DevTools, save a log — just to start understanding the protocol

---

## 12. Suggested repo structure

```
CatanBot/
├── plan.md
├── README.md
├── pyproject.toml            # or requirements.txt
├── src/
│   └── catanbot/
│       ├── __init__.py
│       ├── smoke.py
│       ├── agents/           # all brains, one decide() interface
│       │   ├── base.py
│       │   ├── random_agent.py
│       │   ├── heuristic.py
│       │   ├── alphabeta.py
│       │   └── rl_agent.py
│       ├── env/
│       │   ├── encoding.py    # state<->tensor, action<->id, masks
│       │   └── selfplay_env.py
│       ├── search/            # MCTS (PUCT) + alpha-beta + determinization
│       ├── nn/                # policy/value network (multi-head)
│       ├── train/             # PPO loop, AlphaZero loop, actor/learner
│       ├── arena/             # round-robin, Elo, regression gate
│       └── live/              # colonist.io bridge
│           ├── sniffer.py     # WS capture
│           ├── protocol.py    # JSON <-> internal state
│           ├── executor.py    # Playwright/canvas clicks
│           └── bridge.py      # ties it together
├── scripts/                   # CLI entry points
├── notebooks/                 # analysis & plots
├── data/                      # self-play games, checkpoints (gitignore!)
└── tests/
```

---

## 13. Resources

**Primary framework**
- Catanatron — repo: https://github.com/bcollazo/catanatron
- Catanatron docs: https://docs.catanatron.com
- Catanatron Discord: https://discord.gg/FgFmb75TWd

**Must-read lessons / prior art**
- "5 Ways NOT to Build a Catan AI" (Catanatron author): https://medium.com/@bcollazo2010/5-ways-not-to-build-a-catan-ai-e01bc491af17
- "Learning To Play Settlers of Catan With Deep RL" (settlers-rl): https://settlers-rl.github.io/
- "Modeling Catan through self-play" (Justin Asher): https://justinasher.me/catan_ai
- QSettlers — Deep RL for Catan: https://akrishna77.github.io/QSettlers/
- "Playing Catan with Cross-dimensional Neural Network" (beats jSettler): https://www.researchgate.net/publication/343710996_Playing_Catan_with_Cross-dimensional_Neural_Network

**Reference / alternative env**
- kvombatkere/Catan-AI (the repo you found — good for reading, slower than Catanatron): https://github.com/kvombatkere/Catan-AI

**colonist.io interface**
- "Abusing my computer science knowledge to cheat at Catan" (WS reverse-engineering write-up): https://medium.com/@alberttheblacksheep/abusing-my-computer-science-knowledge-to-cheat-at-catan-a0f72fa30309
- colonist.io: https://colonist.io/

**RL background**
- AlphaZero (Science paper): https://www.science.org/doi/10.1126/science.aar6404
- CleanRL (single-file PPO, easy to hack masking + multi-head): https://github.com/vwxyzjn/cleanrl
