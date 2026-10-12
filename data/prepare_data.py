"""Rebuild the exact train/val/test splits used for the reported results.

    python data/prepare_data.py --data-root datasets

Downloads the six UEA archives, rebuilds the pooled case set, and writes the
layout phcssm.trainer.load_uea_pretreated_jax reads. The partition comes from
the index files in data/splits/, so the result is the partition behind Table 1
rather than a fresh random one.

Needs only numpy and the standard library. About 820 MB is downloaded once and
cached under --download-dir, and the written splits take about 470 MB.

The pool follows Walker et al. (2024). Original train and test cases are
concatenated in archive order, exact duplicate cases are dropped keeping the
first occurrence, and the surviving order is the archive order. Only EigenWorms
loses anything, 23 of its 259 cases. Each index file carries a checksum of the
pool it was computed against, verified before any split is written, so an
archive that has changed fails loudly instead of yielding wrong data.
"""
import argparse
import hashlib
import io
import json
import os
import pickle
import sys
import urllib.request
import zipfile

import numpy as np

BASE = "http://www.timeseriesclassification.com/aeon-toolkit/%s.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(HERE, "splits")
DATASETS = ["Heartbeat", "SelfRegulationSCP1", "SelfRegulationSCP2",
            "EthanolConcentration", "MotorImagery", "EigenWorms"]


def log(msg):
    sys.stdout.write("%s\n" % msg)
    sys.stdout.flush()


def fetch(name, download_dir):
    """Download and extract the two combined arff files for one dataset."""
    os.makedirs(download_dir, exist_ok=True)
    zpath = os.path.join(download_dir, "%s.zip" % name)
    wanted = ["%s_TRAIN.arff" % name, "%s_TEST.arff" % name]
    out = os.path.join(download_dir, name)
    paths = [os.path.join(out, w) for w in wanted]
    if all(os.path.isfile(p) for p in paths):
        return paths
    if not os.path.isfile(zpath):
        log("  downloading %s" % name)
        urllib.request.urlretrieve(BASE % name, zpath)
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        members = z.namelist()
        for want, path in zip(wanted, paths):
            member = next((m for m in members if m.endswith(want)), None)
            if member is None:
                raise RuntimeError("%s not found inside %s" % (want, zpath))
            with z.open(member) as src, io.open(path, "wb") as dst:
                dst.write(src.read())
    return paths


def parse_arff(path):
    """Read a UEA multivariate relational arff.

    One case per data line. The relational value sits inside single quotes,
    dimensions separated by a literal backslash-n and values comma separated
    within a dimension. The class label follows the closing quote.
    """
    cases, labels = [], []
    with io.open(path, encoding="utf-8", errors="replace") as handle:
        in_data = False
        for line in handle:
            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                continue
            line = line.rstrip("\r\n")
            if not line or line.startswith("%"):
                continue
            start, end = line.find("'"), line.rfind("'")
            if start < 0 or end <= start:
                continue
            dims = line[start + 1:end].split("\\n")
            rows = [np.array(d.split(","), dtype=np.float64) for d in dims]
            cases.append(np.stack(rows).T)          # (timesteps, channels)
            labels.append(line[end + 1:].lstrip(",").strip())
    if not cases:
        raise RuntimeError("no cases parsed from %s" % path)
    return cases, labels


def build_pool(name, download_dir):
    """The deduplicated pool in archive order, float32, with integer labels."""
    train_file, test_file = fetch(name, download_dir)
    train_cases, train_labels = parse_arff(train_file)
    test_cases, test_labels = parse_arff(test_file)

    data = np.stack(train_cases + test_cases).astype(np.float32)
    classes = sorted(set(train_labels))
    encode = {c: i for i, c in enumerate(classes)}
    labels = np.array([encode[l] for l in train_labels + test_labels],
                      dtype=np.int32)

    before = len(data)
    _, first = np.unique(data, axis=0, return_index=True)
    keep = np.sort(first)
    data, labels = data[keep], labels[keep]
    if before != len(data):
        log("  dropped %d duplicate cases" % (before - len(data)))
    return data, labels, classes


def pool_sha1(data):
    return hashlib.sha1(
        np.ascontiguousarray(data, dtype=np.float32).tobytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True,
                        help="where to write <Dataset>/<seed>/{X,y}_{split}.pkl")
    parser.add_argument("--download-dir", default="uea_download",
                        help="cache for the downloaded archives")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        choices=DATASETS, metavar="NAME")
    args = parser.parse_args(argv)

    failed = []
    for name in args.datasets:
        index_path = os.path.join(SPLITS, "%s.json" % name)
        if not os.path.isfile(index_path):
            log("%s: no split index at %s" % (name, index_path))
            failed.append(name)
            continue
        with io.open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)

        log(name)
        data, labels, classes = build_pool(name, args.download_dir)

        if len(data) != index["pool_size"]:
            log("  pool holds %d cases, the index expects %d"
                % (len(data), index["pool_size"]))
            failed.append(name)
            continue
        digest = pool_sha1(data)
        if digest != index["pool_sha1"]:
            log("  pool checksum %s does not match the recorded %s"
                % (digest[:12], index["pool_sha1"][:12]))
            log("  the archive has changed, so the recorded partition cannot "
                "be applied to it")
            failed.append(name)
            continue
        log("  pool %d cases, shape %s, checksum ok"
            % (len(data), tuple(data.shape[1:])))

        n_classes = len(classes)
        for seed in sorted(index["seeds"]):
            splits = index["seeds"][seed]
            out_dir = os.path.join(args.data_root, name, seed)
            os.makedirs(out_dir, exist_ok=True)
            for split in ("train", "val", "test"):
                sel = np.array(splits[split], dtype=np.int64)
                onehot = np.zeros((len(sel), n_classes), dtype=np.float32)
                onehot[np.arange(len(sel)), labels[sel]] = 1.0
                for stem, arr in (("X_%s" % split, data[sel]),
                                  ("y_%s" % split, onehot)):
                    path = os.path.join(out_dir, "%s.pkl" % stem)
                    with io.open(path, "wb") as handle:
                        pickle.dump(np.ascontiguousarray(arr), handle,
                                    protocol=pickle.HIGHEST_PROTOCOL)
            log("  seed %s  %s" % (seed, "/".join(
                str(len(splits[s])) for s in ("train", "val", "test"))))

    log("")
    if failed:
        log("failed: %s" % ", ".join(failed))
        return 1
    log("wrote %d dataset(s) under %s" % (len(args.datasets), args.data_root))
    log("the download cache at %s is no longer needed" % args.download_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
