"""Turn patching shards into per-family effects and the summaries plotted in Figures 3 and 4.

For every patched action chunk, reconstruct grasp-site trajectories (all starting from the
recipient's state) and compute the normalized trajectory shift

    E = <tau_patch - tau_r, tau_ref - tau_r> / ||tau_ref - tau_r||^2

against two references:
  fk_object_flip  the recipient scene with the instruction for the other object (backbone patches,
                  Figure 4)
  fk_donor        the clean donor trajectory (interface patches, Figure 3)
Pairs whose reference contrast has RMS below 1 mm are dropped. Effects are averaged over the
directed pairs and seeds within each object-pair family ("family effects"); summaries weight
families equally, with 95% intervals from 5,000 bootstrap resamples of families.

Usage:
  python -m vla_patching.analysis --shards runs/shards --out results
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import MODELS, load_manifest, task_spec
from .fk import FK, projection

METRICS = ["fk_object_flip", "fk_donor", "fk_interface", "native_object_flip", "native_donor"]
FAMILY_KEYS = ["model", "family", "comparison", "kind", "shape", "group", "condition", "layer"]
SUMMARY_KEYS = ["model", "comparison", "kind", "shape", "group", "condition", "layer"]


def csv_save(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def expected_conditions(m, name, smoke=False):
    nl = m["models"][name]["n_layers"]
    layers = m["models"][name]["residual_layers"]
    ii = [("all", list(range(nl)))] + [(f"L{i:02d}", [i]) for i in range(nl)]
    ii += [(f"W{i:02d}_{i + 2:02d}", list(range(i, i + 3))) for i in range(nl - 2)]
    if smoke:
        keep = {"all", "L00", f"L{nl // 2:02d}", f"L{nl - 1:02d}", "W00_02",
                f"W{nl // 2:02d}_{nl // 2 + 2:02d}", f"W{nl - 3:02d}_{nl - 1:02d}"}
        ii = [(k, v) for k, v in ii if k in keep]
        layers = sorted({0, nl // 2, layers[-1]})
    return {"vlm": [(f"{g}_L{i:02d}", [i], g) for g in m["groups"] for i in layers],
            "interface": [(k, v, None) for k, v in ii]}


def collect(m, shards, smoke=False, strict=False, measurements_path=None):
    """Compute E for every patch in every complete shard; return per-family effect rows."""
    fk = FK()
    values = defaultdict(lambda: defaultdict(list))
    missing, total = [], 0
    writer = raw = None
    if measurements_path:
        raw = open(measurements_path, "w", newline="")
    for task in range(3 * 13 * len(m["seeds"])):
        name, fi, seed = task_spec(task)
        f = m["families"][fi]
        folder = Path(shards) / name / f"family{fi:02d}" / f"seed{seed}"
        status = folder / "status.json"
        if not status.exists() or json.loads(status.read_text())["state"] != "complete":
            missing.append(task)
            if strict:
                raise RuntimeError(f"Incomplete shard {folder}")
            continue
        validation = json.loads((folder / "validation.json").read_text())
        assert all(validation[k] for k in ["production_exact", "live_tokens_exact", "matched_noise",
                                           "interface_recipient_vlm_exact"])
        assert all(validation["self_patch"].values())
        assert all(v["reference_exact"] for k, v in validation["full_transplant"].items() if k.startswith("interface:"))
        clean = {int(k): np.asarray(v, np.float64) for k, v in json.loads((folder / "clean.json").read_text()).items()}
        references = {k: np.asarray(v, np.float64) for k, v in json.loads((folder / "references.json").read_text()).items()}
        conds = expected_conditions(m, name, smoke)
        count = 0
        for scenario, pairs in f["pairs"].items():
            for d, t in pairs:
                o = m["observations"][str(t)]
                state = o["state"]
                b = fk.trajectory(clean[t], name, state)
                donor = fk.trajectory(clean[d], name, state)
                other = fk.trajectory(clean[o["other"]], name, state)
                reference = fk.trajectory(references[f"{d}>{t}"], name, state)
                axes = {"fk_object_flip": other - b, "fk_donor": donor - b, "fk_interface": reference - b,
                        "native_object_flip": clean[o["other"]][:, :7] - clean[t][:, :7],
                        "native_donor": clean[d][:, :7] - clean[t][:, :7]}
                for kind, conditions in conds.items():
                    p = folder / "results" / f"{scenario}_{d}_{t}_{kind}.json"
                    expected = [dict(comparison=scenario, donor=d, target=t, kind=kind, condition=k, layers=ls, group=g)
                                for k, ls, g in conditions]
                    assert json.loads(p.read_text())["records"] == expected, p
                    with np.load(p.with_suffix(".npz")) as z:
                        actions = z["actions"].astype(np.float64)
                    assert len(actions) == len(expected) and np.isfinite(actions).all()
                    for r, action in zip(expected, actions):
                        if kind != "vlm" and r["condition"] == "all":
                            assert np.array_equal(action, references[f"{d}>{t}"])
                        if kind == "vlm" and scenario == "B" and r["group"] in ("exo", "wrist") and name != "pi05":
                            assert np.array_equal(action, clean[t])
                        patched = fk.trajectory(action, name, state)
                        shape = ("residual" if kind == "vlm" else "all" if r["condition"] == "all"
                                 else "window3" if len(r["layers"]) == 3 else "single")
                        rec = dict(model=name, family=fi, seed=seed, comparison=scenario, donor=d, target=t, kind=kind,
                                   shape=shape, condition=r["condition"], group=r["group"] or "",
                                   layer=float(np.mean(r["layers"])))
                        key = (name, fi, scenario, kind, shape, rec["group"], r["condition"], rec["layer"])
                        for metric in METRICS:
                            delta = patched - b if metric.startswith("fk") else action[:, :7] - clean[t][:, :7]
                            value, _, _, _ = projection(delta, axes[metric],
                                                        m["threshold_fk_rms_m"] if metric.startswith("fk") else 0)
                            rec[metric] = value
                            if value is not None:
                                values[key][metric].append(value)
                        if raw is not None:
                            if writer is None:
                                writer = csv.DictWriter(raw, fieldnames=list(rec))
                                writer.writeheader()
                            writer.writerow(rec)
                        count += 1
        total += count
        print(f"shard {task:3d} {name} family {fi:2d} seed {seed}: {count} patches (cumulative {total})", flush=True)
    if raw is not None:
        raw.close()
    rows = []
    for key, metrics in sorted(values.items()):
        row = dict(zip(FAMILY_KEYS, key))
        for metric in METRICS:
            vv = metrics[metric]
            row[metric] = float(np.mean(vv)) if vv else None
            row[metric + "_n"] = len(vv)
        rows.append(row)
    return rows, missing, total


def summarize(families):
    """Equal-weight family means with 95% family-bootstrap intervals (seeded, deterministic)."""
    groups = defaultdict(list)
    for row in families:
        groups[tuple(row[k] for k in SUMMARY_KEYS)].append(row)
    rows, rng = [], np.random.default_rng(42026)
    for key, vv in sorted(groups.items()):
        row = dict(zip(SUMMARY_KEYS, key))
        for metric in METRICS:
            values = np.asarray([r[metric] for r in vv if r[metric] is not None], float)
            n = len(values)
            row[metric] = float(np.mean(values)) if n else None
            row[metric + "_families"] = n
            ci = np.quantile(np.mean(values[rng.integers(n, size=(5000, n))], axis=1), [.025, .975]) if n else [None, None]
            row[metric + "_lo"] = float(ci[0]) if n else None
            row[metric + "_hi"] = float(ci[1]) if n else None
        rows.append(row)
    return rows


def interface_bc(families):
    """Average comparisons B and C within each family (both required) before summarizing (Figure 3)."""
    grouped = defaultdict(list)
    for r in families:
        if r["kind"] in ("interface", "joint") and r["comparison"] in ("B", "C"):
            grouped[tuple(r[k] for k in ["model", "family", "kind", "shape", "group", "condition", "layer"])].append(r)
    bc = []
    for key, v in grouped.items():
        rec = dict(zip(["model", "family", "kind", "shape", "group", "condition", "layer"], key), comparison="BC")
        for metric in METRICS:
            vals = [r[metric] for r in v if r[metric] is not None]
            rec[metric] = float(np.mean(vals)) if len(vals) == 2 else None
        bc.append(rec)
    return summarize(bc)


def load_family_effects(path):
    """Read family_effects.csv back with the types `collect` produces."""
    rows = []
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            row = dict(model=r["model"], family=int(r["family"]), comparison=r["comparison"], kind=r["kind"],
                       shape=r["shape"], group=r["group"], condition=r["condition"], layer=float(r["layer"]))
            for metric in METRICS:
                row[metric] = float(r[metric]) if r[metric] not in ("", "None") else None
                row[metric + "_n"] = int(r[metric + "_n"])
            rows.append(row)
    return rows


def write_summaries(families, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    csv_save(out / "summary.csv", summarize(families))
    csv_save(out / "interface_bc_summary.csv", interface_bc(families))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default="runs/shards", help="directory written by vla_patching.runner")
    ap.add_argument("--out", default="results")
    ap.add_argument("--smoke", action="store_true", help="analyze shards written with runner --smoke")
    ap.add_argument("--strict", action="store_true", help="fail if any of the 312 shards is incomplete")
    ap.add_argument("--measurements", action="store_true", help="also write every per-patch value (large)")
    a = ap.parse_args()
    m = load_manifest()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    families, missing, total = collect(m, a.shards, a.smoke, a.strict,
                                       out / "measurements.csv" if a.measurements else None)
    if not families:
        print("No complete shards found under", a.shards)
        return
    csv_save(out / "family_effects.csv", families)
    write_summaries(families, out)
    print(f"{total} patches from {312 - len(missing)} complete shards -> {out}/family_effects.csv, summary.csv, "
          f"interface_bc_summary.csv" + (f" ({len(missing)} shards missing)" if missing else ""))


if __name__ == "__main__":
    main()
