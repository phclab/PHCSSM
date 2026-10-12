# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright National Yang Ming Chiao Tung University, Taiwan.
# Creator Po-Han Chiang. Patent pending.
# Noncommercial use only. For commercial licensing see LICENSE.
"""Stage two of the protocol: train one configuration on the five reporting seeds.

    python train.py --config configs/scp2.json --data-root /path/to/datasets

With no --seeds all five splits run and the mean and sample standard deviation
of test accuracy are printed, which is the pair reported in Table 1. Any
setting can be overridden, for example --set neuron_dim=32 --set lr_max=1e-4.
"""
import argparse
import json
import os
import statistics
import sys

from phcssm import config_io, trainer

SEEDS = [2345, 3456, 4567, 5678, 6789]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="path to a configs/*.json file")
    parser.add_argument("--data-root", required=True,
                        help="directory holding <DatasetName>/<seed>/X_train.pkl")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--out", default="runs",
                        help="directory for checkpoints, epoch logs and summary")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    config = config_io.load(args.config, args.data_root, args.overrides)
    os.makedirs(args.out, exist_ok=True)
    name = config["dataset_name"]
    tag = os.path.splitext(os.path.basename(args.config))[0]

    print(f"{name}: neuron_dim {config['neuron_dim']}, "
          f"lr {config['lr_max']}, readout {config['readout_source']}, "
          f"K {config['num_transmissions']}, batch {config['batch_size']}")

    runs = []
    for seed in args.seeds:
        run = dict(config)
        run["_config_filename"] = f"phcssm_{tag}"
        run["save_dir"] = args.out
        train_data, val_data, test_data = trainer.load_uea_pretreated_jax(
            run, name, seed)
        test_acc, val_acc, mstep_train, mstep_test = trainer.train(
            run, seed, train_data, val_data, test_data)
        runs.append({"seed": seed, "val_acc": val_acc, "test_acc": test_acc,
                     "mstep_train": mstep_train, "mstep_test": mstep_test})
        print(f"  seed {seed}: val {val_acc * 100:6.2f}  "
              f"test {test_acc * 100:6.2f}  "
              f"mstep {mstep_test:.1f} of {config['num_transmissions']}")

    accs = [run["test_acc"] * 100 for run in runs]
    mean = statistics.mean(accs)
    sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
    summary = {
        "dataset": name,
        "config": {k: v for k, v in config.items() if k != "data_root"},
        "seeds": args.seeds,
        "runs": runs,
        "mean_test_acc": mean,
        "sd_test_acc": sd,
    }
    path = os.path.join(args.out, f"{tag}_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"{name}: {mean:.1f} +/- {sd:.1f} over {len(accs)} seeds")
    print(f"summary written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
