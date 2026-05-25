# Max batch size by GPU

Probed via `find_max_batch_size.sh` — fresh python subprocess per attempt,
running 2 train + 2 validation steps so every activation, gradient and
optimizer-state tensor is actually allocated.

The row for each `(GPU, mode)` pair is rewritten after every successful
attempt, so the latest known-good survives a crash mid-probe.

| GPU | Total VRAM (MiB) | Mode | Max batch | Peak VRAM at max (MiB) | Probed at |
|---|---|---|---|---|---|
| NVIDIA L40S | 46068 | full | 2 | 38875 | 2026-05-25T09:23:50Z |
| NVIDIA L40S | 46068 | lora | 2 | 26365 | 2026-05-25T09:24:46Z |
