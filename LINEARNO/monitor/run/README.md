# Standard PDE benchmark launchers

Each dataset script trains the paper-configured Transolver, then evaluates
`checkpoints/checkpoint_final.pth`. Both phases run through the optional
propagation-kernel monitor.

Defaults:

- repository root: inferred from the script location
- dataset root: `/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/fno`
- physical GPU: `1`
- epochs: `500`
- monitor cadence: validation 1, 50, 100, ...
- monitored samples per snapshot: `4`
- sampled point rows per kernel axis: `4096`

Environment variables can override these values: `DATA_ROOT`, `GPU_ID`,
`EPOCHS`, `SEED`, `OUTPUT_ROOT`, `CHECKPOINT_INTERVAL`,
`VISUALIZATION_INTERVAL`, `CAPTURE_EVERY_VALIDATIONS`,
`MONITOR_MAX_SAMPLES`, `MONITOR_MAX_POINTS`, and `MONITOR_SEED`.

Run one dataset, for example:

```bash
bash LINEARNO/monitor/run/darcy.sh
```

Normal experiment outputs remain under
`PDE-Solving-StandardBenchmark/output`. Monitor-only outputs are written to
`LINEARNO/monitor/output/<dataset>/<timestamp>_<train-or-eval>/`.
