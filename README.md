# PHCSSM

Reference implementation for

> Po-Han Chiang. **Parallelized Hierarchical Connectome: A Spatiotemporal
> Recurrent Framework for Spiking State-Space Models.** *Array* (Elsevier),
> 2026, in press.

National Yang Ming Chiao Tung University, Taiwan.

PHCSSM is a spiking state-space model carrying five biological constraints
inside a fully parallelizable recurrence. The constraints are ALIF dynamics,
lateral and feedback connectivity, Tsodyks-Markram short-term plasticity,
Dale's Law with an E/I-asymmetric topology, and STDP.

The temporal recurrences stay diagonal in the neuron index and are evaluated by
a log-domain prefix scan in O(log T) work-depth. Spatial coupling is settled on
a second axis, the within-timestep transmission loop, which runs for a fixed
`num_transmissions` iterations. That loop is a fixed-point solver over the
iteration index rather than an evolution in time. Every iteration recomputes the
whole timeline from the same initial state, and only the loop input accumulates
synaptic feedback, so learnable lateral weights sit inside the recurrence
without costing the parallel scan.

## Layout

```
phcssm/        the model, the training loop, configuration loading
train.py       stage two of the protocol, five-seed reporting
grid_search.py stage one, the 24-configuration single-seed search
configs/       the six selected configurations behind Table 1
data/          the preparation script and the recorded train/val/test partition
rsnn/          the sequential RSNN execution mode and the two-mode comparison
stm32/         the Cortex-M4F C port and its weight exporter
```

## Install

```bash
pip install -r requirements.txt
```

The pins are jax 0.10.0, jaxlib 0.10.0, equinox 0.13.7, optax 0.2.3,
numpy 2.4.4 and tqdm 4.66.5 on Python 3.11. A GPU run needs the CUDA build of
the same jax version.

## Data

The benchmark is the six-dataset subset of the UEA Multivariate Time Series
Classification Archive (Bagnall et al. 2018) used by Walker et al. (2024).

| Dataset | Config | T | Channels | Classes |
|---|---|---|---|---|
| Heartbeat | `configs/heart.json` | 405 | 61 | 2 |
| SelfRegulationSCP1 | `configs/scp1.json` | 896 | 6 | 2 |
| SelfRegulationSCP2 | `configs/scp2.json` | 1,152 | 7 | 2 |
| EthanolConcentration | `configs/ethanol.json` | 1,751 | 3 | 4 |
| MotorImagery | `configs/motor.json` | 3,000 | 64 | 2 |
| EigenWorms | `configs/worms.json` | 17,984 | 6 | 5 |

### Prepare the data

```bash
python data/prepare_data.py --data-root datasets
```

That fetches the six archives from the UEA website, rebuilds the pooled case
set, and writes the partition the reported results were trained on. Around
820 MB is downloaded once and cached under `--download-dir`, the written splits
take about 470 MB, and nothing beyond numpy and the standard library is needed.
Pass `--datasets` to do a subset.

`data/splits/` records that partition explicitly, one index file per dataset and
52 KB in total, because the permutation that produced it cannot be regenerated
from a seed. Each index carries a checksum of the pool it was computed against,
and the script verifies it before writing anything, so an archive that has
changed fails loudly instead of yielding different data under the same name.

### The split protocol

The partition follows Walker et al. (2024). The original train and test cases
of a dataset are pooled in archive order, exact duplicate cases are dropped
keeping the first occurrence, and the remainder is split 70/15/15, once per
seed, over the five seeds 2345, 3456, 4567, 5678 and 6789. The bounds are
`int(N * 0.70)` and `int(N * 0.85)`.

Deduplication changes only EigenWorms, which loses 23 of its 259 cases. These
are the resulting sizes, which also check a copy of the data obtained some
other way.

| Dataset | UEA cases | Pool | Train | Val | Test |
|---|---|---|---|---|---|
| Heartbeat | 409 | 409 | 286 | 61 | 62 |
| SelfRegulationSCP1 | 561 | 561 | 392 | 84 | 85 |
| SelfRegulationSCP2 | 380 | 380 | 266 | 57 | 57 |
| EthanolConcentration | 524 | 524 | 366 | 79 | 79 |
| MotorImagery | 378 | 378 | 264 | 57 | 57 |
| EigenWorms | 259 | 236 | 165 | 35 | 36 |

### On-disk layout

`prepare_data.py` writes this, so it is only needed to use data prepared some
other way. `phcssm.trainer.load_uea_pretreated_jax` expects one directory per
dataset and one subdirectory per seed.

```
<data-root>/<DatasetName>/<seed>/X_train.pkl
<data-root>/<DatasetName>/<seed>/y_train.pkl
<data-root>/<DatasetName>/<seed>/X_val.pkl
<data-root>/<DatasetName>/<seed>/y_val.pkl
<data-root>/<DatasetName>/<seed>/X_test.pkl
<data-root>/<DatasetName>/<seed>/y_test.pkl
```

`<DatasetName>` is the `dataset_name` of the configuration, which is the archive
name as spelled in the table above. Each `X` pickle holds an array of shape
`(cases, timesteps, channels)` and each `y` pickle holds one-hot labels of shape
`(cases, classes)`, which the loader reduces with argmax.

Some preprocessing pipelines prepend a time index as channel 0. The loader drops
the leading channel whenever a stored array carries exactly one channel more
than the `input_dim` of the configuration, so either form works.

## Run one dataset

Stage two trains a selected configuration on the five reporting seeds and prints
the mean and sample standard deviation of test accuracy.

```bash
python train.py --config configs/scp2.json --data-root /path/to/datasets
```

Restrict the seeds with `--seeds`, redirect checkpoints and logs with `--out`,
and override any entry of the configuration with `--set`.

```bash
python train.py --config configs/heart.json --data-root /path/to/datasets \
    --seeds 2345 --out runs
```

Per seed it writes a checkpoint and an epoch log under `--out`, named after the
configuration file, and one summary for the whole run. The accuracies in the
checkpoint name are the best validation and the final test score, and the
trailing number is the best epoch, so that command produces

```
runs/phcssm_heart_Seed2345_77.05_75.81_e799.pth
runs/phcssm_heart_Seed2345_epoch_log.csv
runs/heart_summary.json
```

## Reproduce Table 1

Both stages train at the same budget, 10,000 epochs with early-stopping
patience 1,000.

**Stage one.** A single-seed grid search, 24 configurations, selecting the
highest validation accuracy.

| Axis | Values |
|---|---|
| Learning rate | 1e-3, 1e-4 |
| Neuron dim | 16, 32, 64 |
| Readout source | voltage, spike |
| Topology mask | 0 feedforward, 1 bidirectional |

```bash
python grid_search.py --config configs/scp2.json --data-root /path/to/datasets
```

Runs are cached under `--out`, so an interrupted sweep resumes. The leaderboard
is ordered by validation accuracy and names any tie at the top together with the
command that resolves it, which is validation accuracy under a second seed.

Stage one does not score the test split. Validation accuracy is the only
accuracy it computes, prints or caches.

The configurations already in `configs/` are the ones this stage selected, so
run stage one to reproduce the selection and skip it to go straight to the
reported numbers.

**Stage two.** Run `train.py` on the winner for all five seeds. Repeating that
for the six configurations reproduces the PHCSSM row of Table 1.

| Dataset | Learning rate | Neuron dim | Readout | Mask | Test accuracy | Parameters |
|---|---|---|---|---|---|---|
| Heartbeat | 1e-4 | 16 | voltage | 0 | 73.9 +/- 3.9 | 1,312 |
| SelfRegulationSCP1 | 1e-3 | 64 | voltage | 0 | 80.7 +/- 1.6 | 3,388 |
| SelfRegulationSCP2 | 1e-3 | 64 | voltage | 1 | 58.3 +/- 6.3 | 4,877 |
| EthanolConcentration | 1e-3 | 64 | voltage | 1 | 31.1 +/- 6.1 | 4,673 |
| MotorImagery | 1e-4 | 16 | voltage | 1 | 54.7 +/- 4.5 | 1,444 |
| EigenWorms | 1e-3 | 64 | voltage | 1 | 85.0 +/- 5.8 | 4,891 |

Accuracy is the mean over the five seeds. The parameter column counts trainable
parameters with the recurrent weight taken at its unmasked support, since the
entries the topology mask zeroes carry no gradient.

The mask column is the `hierarchy_matrix` of the configuration, which
`grid_search.py` tags `hm0` and `hm1`.

## The sequential RSNN mode

The same trained weights run in a second execution mode, one timestep at a time,
which is the mode the paper deploys. `rsnn/compare_modes.py` runs both on a
checkpoint and reports the agreement.

```bash
python train.py --config configs/heart.json --data-root /path/to/datasets \
    --seeds 2345 --out runs
python rsnn/compare_modes.py runs/phcssm_heart_Seed2345_*.pth
```

## Deployment on a Cortex-M4F

`stm32/` holds the fp32 C port, its PlatformIO project, and the exporter that
turns a checkpoint into a weight header.

```bash
python stm32/export_weights.py runs/phcssm_heart_Seed2345_*.pth \
    -o stm32/src/weights/rsnn_weights_heart_seed2345.h
```

One weight header is included as a worked example, the Heartbeat seed 2345 one
that `stm32/src/rsnn_config.h` names by default, so the firmware builds without
training first.

```bash
cd stm32 && pio run
```

To build another target, export its weights to the path the environment expects
and name that environment. There is one `stm32_<dataset>_seed<seed>` environment
per cell of Table 1, and each expects
`src/weights/rsnn_weights_<dataset>_seed<seed>.h`.

```bash
python export_weights.py ../runs/phcssm_scp1_Seed2345_*.pth \
    -o src/weights/rsnn_weights_scp1_seed2345.h
pio run -e stm32_scp1_seed2345
```

Outside PlatformIO the same selection is a define, so pass
`-DRSNN_VARIANT_INC='"weights/rsnn_weights_scp1_seed2345.h"'` to the compiler,
or edit the default in `src/rsnn_config.h` before an Arduino IDE build.

## License

PolyForm Noncommercial 1.0.0. Noncommercial use is permitted, and educational
institutions and public research organizations are covered regardless of the
source of their funding. Commercial use is not licensed here. See `LICENSE`,
which carries the contact for commercial licensing.
