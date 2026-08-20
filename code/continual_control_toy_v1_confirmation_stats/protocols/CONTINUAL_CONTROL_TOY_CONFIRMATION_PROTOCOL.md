# Single-Agent Continual-Control Confirmation Protocol

Frozen before running any confirmation seed: 2026-07-31.

The excluded development experiment used seeds `5000..5031` and configuration
SHA-256 `4546d16604b47ec81b7443d9666576c5e93f929ed23a44fa2eff74945b3ef19c`.
All development integrity and direction gates passed. The strongest registered
simple prediction baseline on development was `return_slope`.

Confirmation uses untouched seeds `0..63`. It inherits the development plant,
drift distribution, online estimator, reward, candidate, thresholds, horizon,
controllers, and intervention costs through
`configs/continual_control/v1_confirmation.json`. No simulator or decision
parameter may change.

The two co-primary paired contrasts are:

1. per-stream average precision of structural risk minus return slope;
2. per-stream mean reward of the certificate controller minus the
   return-trigger controller.

Each interval is a 95% percentile bootstrap interval obtained by resampling
the 64 complete uninterrupted streams. Checkpoints and drift incidents are
never resampled as independent units. The joint confirmation claim is accepted
only when the lower endpoint is above zero for both co-primary contrasts. This
is an intersection-union rule: both component claims must pass at level 0.05.

All registered secondary predictors, intervention arms, integrity checks, and
effect estimates are reported regardless of direction. Confirmation artifacts
are written separately from development artifacts.
