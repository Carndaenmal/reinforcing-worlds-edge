# Single-Agent Continual-Control Development Protocol

Frozen before inspecting development results: 2026-07-31. The robust
disturbance formulation below was amended after an unregistered engineering
smoke seed (`999999`) showed that a deterministic finite-horizon definition was
flat until nominal instability. No registered development seed was run or
inspected before this amendment.

## Scope

This is an independent construction-level validation track. It does not replace
the peer-learning claims or the LBF experiments. It asks whether a
core-compatible structural monitor can prospectively detect loss of safe
closed-loop behavior in a single-agent, genuinely continuous stream.

Each independent seed produces one uninterrupted trajectory. The physical
state, controller, online system-identification state, rolling monitors, and
intervention state are never reset at a drift boundary. There are no episodes.

## Dynamical system and continual learning

The scalar plant is

```text
x[t+1] = a*x[t] + b[t]*u[t] + w[t].
```

The action is continuous and clipped. The nominal controller is inexpensive
but loses stability when actuator authority `b[t]` becomes small. A more
aggressive backup controller remains stable in the registered drift range and
pays a fixed intervention cost. Small, seed-fixed probing actions allow each
treatment to update a scalar recursive-least-squares estimate of `b[t]` from
its own transition stream. Neither the true gain nor future drift is provided
to the trigger.

The actuator gain follows a continuous, piecewise-linear high-to-low-to-high
schedule. Event start times, ramp lengths, plateau lengths, and low gains are
sampled once from frozen distributions using the independent stream seed.
There are at least four drift-and-recovery events in every stream.

## Pre-specified structural quantity

For a uniform reference set `x in [-R, R]` and the registered bounded
disturbance set `w in [-d_core, d_core]`, the nominal controller's estimated
one-step robust viability coverage is

```text
c_hat[t] = min(1, (R - d_core) /
                     (R*|a - b_hat[t]*k_nominal|)).
```

with the zero closed-loop case defined as one. This is exactly the fraction of
the reference interval guaranteed to remain in that interval for every
registered disturbance. The exact diagnostic substitutes the latent `b[t]`
for `b_hat[t]`; it is not an operational predictor. The primary predictor is
risk `1 - c_hat[t]`.

This quantity is fixed independently of reward and state magnitude. Its
scientific purpose is to expose erosion of the reference viable set before an
initially central state has accumulated enough unstable dynamics to reduce
reward.

## Prospective target and baselines

At fixed checkpoints, an observation is at risk only when the nominal
trajectory's current rolling reward is at least the registered healthy
threshold. The positive label is a new downward crossing of the registered
failure threshold within the next registered horizon. Existing failure
plateaus and incomplete tail horizons are censored. Recovery re-enters the
risk set.

Primary prediction comparison:

```text
1 - estimated viability coverage
    versus
the strongest of current rolling return, return slope, absolute state,
transition-surprise CUSUM, checkpoint time, and a seed-fixed random score.
```

Metrics are average precision, Brier score, false alarms, and structural lead
time. Complete streams—not checkpoints—are resampled in paired bootstrap
intervals.

## Intervention comparison

All treatments receive the same drift schedule, process noise, and probing
sequence within a seed. Each treatment nevertheless evolves its own unbroken
state and online estimate.

- `never`: always use the nominal controller;
- `certificate`: use the backup controller under frozen coverage hysteresis;
- `return_trigger`: use the backup under frozen rolling-return hysteresis;
- `random_matched`: a seed-fixed cyclic shift of the certificate intervention
  schedule, preserving its intervention count;
- `oracle`: use the backup when exact viability coverage crosses the registered
  threshold.

Primary mitigation contrasts are certificate minus never and certificate
minus return-trigger cumulative reward, including control and intervention
costs. Switch count and time in backup mode are reported.

## Data separation and decisions

Development stream seeds are `5000..5031`; they are permanently excluded from
confirmation. Confirmation seeds are `0..63` and are not launched by the
development workflow.

The registered simulator constants, targets, thresholds, predictors, and
primary contrasts are stored in
`configs/continual_control/v1_development.json`. Development may diagnose an
implementation or construct failure, but no seed, checkpoint, or drift event
may be removed based on its result. Any substantive redesign receives a new
version and new excluded development seeds.

Confirmation is eligible only if all integrity gates pass and development
directions are positive for both:

- prediction AP of structural risk minus the strongest simple baseline;
- cumulative return of the certificate trigger minus the return trigger.

These are development direction gates, not confirmatory significance claims.
No final or confirmation job is automatically submitted.
