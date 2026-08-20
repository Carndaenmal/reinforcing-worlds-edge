# Continual cue-MNIST V4 supervised confirmation

This compact artifact contains the registered supervised protocol and
configuration snapshot, and all 64
complete uninterrupted supervised confirmation-stream records used for E12.
The registered supervised confirmation passed both co-primary endpoints.

The manuscript-facing aggregate is
`confirmation_summary_supervised.json`. The original per-stream JSON records
are preserved without modification in `supervised_per_stream_records.zip`.
They can be extracted with:

```bash
unzip supervised_per_stream_records.zip -d supervised_per_stream_records
```

The source archive supplied for incorporation was
`continual_cue_mnist_v4_confirmation_stats_2026-08-01.zip`, SHA-256
`7e38eb4beb6d67c6b8218d25276258e90776af718854dd5ff49820c413639527`.
The supervised-record archive has SHA-256
`2572de4bb2334be54a409c9e7469f8400c1d553acea9a62b6cd0625fb173a211`.

Regenerate the manuscript figure from the aggregate statistics with:

```bash
python3 code/plot_cue_mnist_confirmation.py
```
