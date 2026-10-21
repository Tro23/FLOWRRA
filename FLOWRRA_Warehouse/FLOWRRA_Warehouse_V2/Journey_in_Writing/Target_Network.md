# The target network: how it is chosen and used

## Why there are two networks

The learner improves its estimate `Q(s, a)` towards a **target**:

```
target = reward + gamma * (value of the best next action)
```

If the same network that is being updated also produced that target, every
update would move the thing it is aiming at. The network chases its own tail and
values can run away. So a second copy -- the **target network** -- is held still
and used to compute targets, while the **online network** (`policy_net`) learns.

## How it is created

In `GNNAgent.__init__` (`agent_warehouse.py`):

1. `self.target_net` is built as a second `GNNPolicy`, **identical in
   architecture** to `self.policy_net`.
2. `self.target_net.load_state_dict(self.policy_net.state_dict())` makes it an
   **exact copy** of the online network at the start.
3. `self.target_net.eval()` puts it in evaluation mode for good: its dropout
   (0.1 on the attention weights) is **off**, so targets are deterministic.

It is never trained: no optimiser touches it, and it is only ever run inside
`torch.no_grad()`.

## How it is kept in step

It is refreshed by a **hard copy**, not a gradual blend. In
`main_runner_warehouse.py`, once the buffer holds a full batch, every
environment step does one learning step; a counter of learning steps triggers
`agent.update_target_network()` every `--target-sync` steps (default **1,000**):

```python
if len(agent.memory) >= agent.batch_size:
    agent.learn(...)
    learn_steps += 1
    if learn_steps % args.target_sync == 0:
        agent.update_target_network()      # target_net <- exact copy of policy_net
```

An episode is at most 780 steps, so the target is refreshed roughly **every 1.3
episodes**. The counter runs across episodes; it is not reset per episode.

## How it is used in a learning step

Inside `GNNAgent.learn()`:

1. **Online pass on the current states** gives `Q(s, a)` for every head, with
   gradient. This is what gets updated.
2. **Target pass on the next states**, without gradient, gives the next-state
   values per head (`next_q_per_head`) and their weighted sum (`next_q_sum`).
3. **Mask impossible actions.** Actions the action mask forbids in the next state
   are set to a large negative sentinel before choosing, so the target can
   never be built from an action no fleet could actually take (the fix for the
   cold_run5 divergence).
4. **Choose the next action**, then **score it** -- see below.
5. `target = reward + gamma * score * (1 - done)`, per head.
6. Huber (smooth-L1) loss between `Q(s, a)` and the target, per head, over
   active fleets; gradients clipped to norm 1.0.

The recovery head has its own target, computed the same way from the target
network's `last_recovery_q`.

### Plain DQN (`double_dqn: False`, every run up to cold_run18)

The **target network both chooses and scores** the next action:

```python
next_q_sum, _, next_q_per_head = self.target_net(next_states)
next_a = masked(next_q_sum).argmax()            # target chooses
score  = next_q_per_head.gather(next_a)         # target scores its own choice
```

Taking the maximum over noisy estimates, then trusting that same maximum, biases
values **upward**. Bootstrapping feeds that bias back into the next target, so it
compounds. That is the pattern seen in both replicate runs: cold_run16's goal
loss went 3 -> 26.7 and snapped back to 1.4; cold_run18's climbed 1.4 -> 6.0 and
was still climbing when the run ended.

### Double DQN (`double_dqn: True`, from cold_run19)

The two jobs are **split between the networks**:

```python
next_q_sum, _, next_q_per_head = self.target_net(next_states)   # target: for scoring
next_q_sum, _, _ = self.policy_net(next_states)                 # online: for choosing
next_a = masked(next_q_sum).argmax()            # ONLINE chooses
score  = next_q_per_head.gather(next_a)         # TARGET scores
```

An action that looks best only because of the online network's noise is unlikely
to be scored highly by the separate target network too, which removes most of
the upward bias. Only the *source of the choice* changes: masking, the argmax,
the gather and the target formula are untouched.

Two details make it correct here:

- **The online network chooses in evaluation mode.** It has dropout, which in
  training mode would randomise every choice. It is switched to `eval()` for
  that one pass and restored to its previous mode straight after.
- **A cached value is protected.** A forward pass stores `last_recovery_q` on the
  network, and `learn()` later reads it for the recovery loss -- from the
  *current* states, with gradient. The extra pass on the *next* states would
  have overwritten it with the wrong states and no gradient, **silently freezing
  the recovery head**. It is saved before the pass and restored after. It is the
  only value a forward pass stores on the network.

## How it was verified

- **Identical networks give an identical loss.** With the online network set equal
  to the target network, Double DQN and plain DQN produce the same loss on the
  same batch, to every printed digit (0.0596919246 both ways). This is the
  check that found the cached-value bug: before the fix, the two differed.
- **Different networks give a different loss**, so the switch is really active.
- **The recovery head still learns**: its weights move during a Double DQN step.
- **Switched off, nothing changes**: with `double_dqn: False` and the other new
  switches off, the code reproduces the reference run exactly.

## If instability persists

Levers not yet used, in order of cost:

1. **Soft (Polyak) target updates** -- blend a small fraction of the online
   weights into the target every step, instead of a full copy every 1,000.
   Smoother targets; one new parameter.
2. **A longer `--target-sync`** -- a stiller target, at the price of slower
   propagation of what has been learned.
3. **Adaptive per-head value normalisation (PopArt)** -- if fixed value scales
   prove not to hold as the policy improves.

## The recovery head (cold_run21)

cold_run21 switched on soft target updates (tau 0.002). The three reward heads
held steady -- value levels around 0.02 (safety), 0.3 (delivery), -0.1
(efficiency) -- but the recovery head's value diverged, roughly doubling every
episode or two from episode 22:

| episode | 17 | 23 | 30 | 37 | 42 |
|---|---|---|---|---|---|
| recovery value | -0.43 | 1.8 | 106 | 2,622 | 32,554 |
| recovery loss | 0.004 | 0.99 | 44.5 | 1,032 | 17,099 |

Its inflated value for "invoke" drove invocations to ~500-640 per episode and
locked the warehouse into permanent warning (wasted -> 0). Its loss, entering
training next to the three heads under one clipped gradient budget, took nearly
all of it -- starving the movement heads (completion 0.90 -> 0.85).

Why this head: it was the only one still on plain DQN; its reward is tiny once
scaled, so bootstrapping dominates its target; and a soft-updated target follows
its own rising estimate on every step, where hard copies froze it between jumps.

**Fix 1 -- Double DQN for the recovery target** (`training.recovery_double_dqn`).
The online net picks the next recovery mode, the target net scores it. The
online values come from the pass the main Double DQN already makes, captured
before the cache is restored -- no extra forward pass.

**Fix 2 -- bound the recovery target** (`training.recovery_value_bound`). Per
step, the recovery head's reward is the holon's coherence (-1..0) plus recovery
events: an invocation costs 4, a resolution pays 6, a preemptive success pays 9.
Allowing two of each per step and the holon scale of 70.3:
(1 + 2 x (4 + 6 + 9)) / 70.3 / (1 - 0.99) = **55.5** -- the largest value it
could ever legitimately hold. The runner computes it from the config; the TD
target is clamped to +/- that. cold_run21 reached nearly 600x the bound.

Verified: switches off -> learning identical to the last digit; Double DQN with
identical networks reproduces the plain recovery loss exactly and differs with
different networks; with target values forced to ~10,000, the recovery loss is
10,993 unbounded and 54 bounded.