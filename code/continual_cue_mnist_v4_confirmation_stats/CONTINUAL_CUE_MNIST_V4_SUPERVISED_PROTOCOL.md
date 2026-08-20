# Continual cue-MNIST V4 supervised confirmation

The design, score, comparator, thresholds, endpoints, seed list, treatment
rules, and analysis were frozen before the 64 supervised confirmation streams
were evaluated. Seeds `8200..8263` are disjoint from all development data.

## Stream and structural quantity

Each independent unit is one complete uninterrupted model/optimizer stream.
The stream contains four randomized shortcut-learning and recovery events;
model, optimizer, and monitoring state are never reset. Monitoring uses three
fixed held-out splits: a 2,000-example core probe, a 500-example shift monitor,
and a 1,000-example performance probe. None participates in training.

For the fixed reference cue, let `B[t]` be the core-probe examples classified
correctly at checkpoint `t`. The success-conditioned `U3` coverage is the
fraction of `B[t]` classified correctly under every cue in the registered
three-cue set. Its operational certificate `C[t]` is the Wilson lower
confidence bound. The same certificate defines baseline eligibility, the
prediction score, and the replay trigger.

## Registered endpoints

The prospective prediction score is `max(0, C[t-3] - C[t])`. The target is a
new old-reference accuracy downcrossing below `0.88` within seven checkpoints,
evaluated only from currently healthy checkpoints at or above `0.92`. The
frozen comparator is cue-label association.

The mitigation endpoint compares whole-stream balanced reward under replay
triggered by `C[t]` with replay triggered by observed clean-accuracy
degradation. Additional frozen controls are oracle, intervention-count-matched
random replay, and never intervening.

The two co-primary contrasts are paired within complete stream: structural
erosion AP minus comparator AP, and core-triggered reward minus
accuracy-triggered reward. Their 95% percentile intervals use 10,000 seeded
nonparametric bootstrap resamples of the 64 complete streams. Checkpoints,
incidents, and images are dependent within-stream observations and are not
promoted to the independent-unit count. Confirmation requires every integrity
and premise gate to pass and both paired lower confidence endpoints to exceed
zero.
