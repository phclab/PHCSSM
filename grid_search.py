"""Stage one of the protocol: single-seed grid search for configuration selection.

    python grid_search.py --config configs/scp2.json --data-root /path/to/datasets

The grid is learning rate x neuron dimension x readout source x topology mask,
24 configurations, and everything else stays at the values in --config. One
seed is trained per configuration, at the same epoch and patience budget used
for reporting, and the leaderboard is ordered by validation accuracy. Each run
is cached under --out, so an interrupted sweep resumes and --only re-runs named
configurations.

The test split is never scored here. Nothing this script prints or caches is
derived from it, so a configuration cannot be chosen on test accuracy.

Ties at the grid-search seed are resolved by validation accuracy under a second
seed. When the top of the leaderboard is tied, re-run the tied tags at the next
reporting seed:

    python grid_search.py --config configs/scp2.json --data-root /data \\
        --seed 3456 --only nd64_lr1e-3_voltage_hm0 nd64_lr1e-3_voltage_hm1
"""
import argparse
import itertools
import json
import os
import sys

from phcssm import config_io, trainer

# Grid axes. The two topology masks are M_topo 0 (feedforward, index 0) and
# M_topo 1 (bidirectional, index 1); entry [i][j] is the projection from region
# j to region i and the four flags are (E->E, I->E, E->I, I->I).
TOPOLOGIES = [
    [[[0, 1, 1, 1], [0, 0, 0, 0]], [[1, 1, 0, 0], [1, 1, 1, 1]]],
    [[[1, 1, 1, 1], [1, 1, 0, 0]], [[1, 1, 0, 0], [1, 1, 1, 1]]],
]

GRID = {
    "neuron_dim": [16, 32, 64],
    "lr_max": [1e-3, 1e-4],
    "readout_source": ["voltage", "spike"],
    "hierarchy_index": [0, 1],
}

GRID_SEED = 2345


def run_tag(point):
    lr = f"{point['lr_max']:.0e}".replace("-0", "-")
    return (f"nd{point['neuron_dim']}_lr{lr}"
            f"_{point['readout_source']}_hm{point['hierarchy_index']}")


def grid_points():
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, values))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="path to a configs/*.json file, used as the base")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=GRID_SEED)
    parser.add_argument("--out", default="grid",
                        help="directory for the per-run cache and the logs")
    parser.add_argument("--only", nargs="+", default=None, metavar="TAG",
                        help="restrict the sweep to these run tags")
    args = parser.parse_args(argv)

    base = config_io.load(args.config, args.data_root)
    tag = os.path.splitext(os.path.basename(args.config))[0]
    cache = os.path.join(args.out, f"{tag}_cache")
    os.makedirs(cache, exist_ok=True)
    name = base["dataset_name"]

    points = [p for p in grid_points()
              if args.only is None or run_tag(p) in args.only]
    print(f"{name}: {len(points)} configurations at seed {args.seed}, "
          f"budget {base['epochs']} epochs / patience "
          f"{base['early_stopping_patience']}")

    # The third return value is the test split. Stage one drops it here and
    # passes evaluate_test=False below, so it is never scored.
    train_data, val_data, _test_split = trainer.load_uea_pretreated_jax(
        base, name, args.seed)

    results = []
    for index, point in enumerate(points, 1):
        name_tag = run_tag(point)
        record_path = os.path.join(cache, f"{name_tag}__seed{args.seed}.json")
        if os.path.exists(record_path) and args.only is None:
            with open(record_path, encoding="utf-8") as handle:
                results.append(json.load(handle))
            print(f"  [{index}/{len(points)}] {name_tag}: cached")
            continue

        run = dict(base)
        run["neuron_dim"] = point["neuron_dim"]
        run["lr_max"] = point["lr_max"]
        run["readout_source"] = point["readout_source"]
        run["hierarchy_matrix"] = TOPOLOGIES[point["hierarchy_index"]]
        run["_config_filename"] = f"grid_{tag}_{name_tag}"
        run["save_dir"] = args.out
        config_io.validate(run)

        _, val_acc, mstep_train, _ = trainer.train(
            run, args.seed, train_data, val_data, None, evaluate_test=False)
        record = {"run_tag": name_tag, "seed": args.seed,
                  "val_acc": val_acc, "mstep_train": mstep_train}
        with open(record_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        results.append(record)
        print(f"  [{index}/{len(points)}] {name_tag}: "
              f"val {val_acc * 100:6.2f}  mstep {mstep_train:.2f}")

    results.sort(key=lambda r: -r["val_acc"])
    print(f"\n{name}, ordered by validation accuracy at seed {args.seed}")
    for rank, record in enumerate(results, 1):
        print(f"  {rank:2d}  {record['run_tag']:<30s} "
              f"val {record['val_acc'] * 100:6.2f}")

    if results:
        best = results[0]["val_acc"]
        tied = [r["run_tag"] for r in results if r["val_acc"] == best]
        if len(tied) > 1:
            print(f"\n{len(tied)} configurations tie at "
                  f"{best * 100:.2f} validation accuracy. Resolve with a "
                  f"second seed:\n  --seed 3456 --only {' '.join(tied)}")
        else:
            print(f"\nselected: {tied[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
