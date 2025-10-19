# src/hparam_v3.py
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np
from tqdm import tqdm

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


def run(cmd, cwd=None, stdout_path=None):
    """Run a shell command, optionally tee stdout to a file. Returns (rc, stdout_text)."""
    env = os.environ.copy()
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out_lines = []
    with open(stdout_path, "w") if stdout_path else nullcontext() as fh:
        for line in p.stdout:
            out_lines.append(line)
            if fh:
                fh.write(line)
    rc = p.wait()
    return rc, "".join(out_lines)


class nullcontext:
    def __enter__(self): return None
    def __exit__(self, *exc): return False


def load_metrics_json(fp: str) -> dict:
    if not os.path.exists(fp):
        return {}
    with open(fp, "r") as f:
        return json.load(f)


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def main(args):
    # ----- Baseline -----
    # Priority: explicit --baseline_dir > reports/analysis/metrics_overall.json > fall back to current run of (L=1, tau=0.03, win=16, stride=4)
    if args.baseline_dir:
        base_metrics = load_metrics_json(os.path.join(args.baseline_dir, "analysis", "metrics_overall.json"))
    else:
        base_metrics = load_metrics_json("reports/analysis/metrics_overall.json")

    # If still empty, do one baseline eval+analyze
    if not base_metrics:
        print("No baseline metrics found. Running a quick baseline (L=1.0, tau=0.03, win=16, stride=4)...")
        base_out = os.path.join(args.out_root, "baseline")
        ensure_dir(base_out)
        # Evaluate
        ev_cmd = [
            sys.executable, "src/eval_temporal_alignment.py",
            "--vid_emb_dir", args.vid_emb_dir,
            "--aud_emb_dir", args.aud_emb_dir,
            "--annotations_csv", args.annotations_csv,
            "--curve_dir", os.path.join(base_out, "curves"),
            "--summary_csv", os.path.join(base_out, "summary.csv"),
            "--tau", "0.03",
            "--L_sec", "1.0",
            "--audio_pick", "middle",
            "--plot_n", "0",
        ]
        rc, _ = run(ev_cmd, stdout_path=os.path.join(base_out, "eval_stdout.txt"))
        if rc != 0:
            print("Baseline eval failed. Aborting.")
            sys.exit(1)
        # Analyze
        an_cmd = [
            sys.executable, "src/analyze_results.py",
            "--summary_csv", os.path.join(base_out, "summary.csv"),
            "--out_dir", os.path.join(base_out, "analysis"),
            "--bins", "10", "--hist_bins", "40",
        ]
        rc, _ = run(an_cmd, stdout_path=os.path.join(base_out, "analyze_stdout.txt"))
        if rc != 0:
            print("Baseline analysis failed. Aborting.")
            sys.exit(1)
        base_metrics = load_metrics_json(os.path.join(base_out, "analysis", "metrics_overall.json"))

    if not base_metrics:
        print("Could not load baseline metrics. Aborting.")
        sys.exit(1)

    base_mae = float(base_metrics.get("mae_mean", np.nan))
    base_hit1 = float(base_metrics.get("hit_1.0_mean", np.nan))
    base_inside = float(base_metrics.get("inside_mean", np.nan))
    print(f"Baseline: MAE={base_mae:.4f}, Hit@1.0={base_hit1:.4f}, Inside={base_inside:.4f}")

    # ----- Sweep space -----
    L_list     = [float(x) for x in args.L_list]
    tau_list   = [float(x) for x in args.tau_list]
    win_list   = [int(x) for x in args.win_list]
    stride_list= [int(x) for x in args.stride_list]

    ensure_dir(args.out_root)
    rows = []
    combos = list(product(L_list, tau_list, win_list, stride_list))

    print(f"Total combos: {len(combos)}  |  results → {args.out_root}")

    for (L_sec, tau, win, stride) in tqdm(combos, desc="HyperV3", unit="cfg"):
        tag = f"L_{L_sec}_tau_{tau}_win_{win}_stride_{stride}"
        run_dir = os.path.join(args.out_root, tag)
        curves_dir = os.path.join(run_dir, "curves")
        analysis_dir = os.path.join(run_dir, "analysis")
        summary_csv = os.path.join(run_dir, "summary.csv")
        ensure_dir(run_dir)

        # 1) Evaluate (reuses cached embeddings; cheap)
        ev_cmd = [
            sys.executable, "src/eval_temporal_alignment.py",
            "--vid_emb_dir", args.vid_emb_dir,
            "--aud_emb_dir", args.aud_emb_dir,
            "--annotations_csv", args.annotations_csv,
            "--curve_dir", curves_dir,
            "--summary_csv", summary_csv,
            "--tau", str(tau),
            "--L_sec", str(L_sec),
            "--audio_pick", "middle",
            "--plot_n", str(args.plot_n),
        ]
        ev_rc, _ = run(ev_cmd, stdout_path=os.path.join(run_dir, "eval_stdout.txt"))
        if ev_rc != 0 or not os.path.exists(summary_csv):
            rows.append({"L_sec": L_sec, "tau": tau, "win": win, "stride": stride,
                         "mae_mean": np.nan, "mae_median": np.nan,
                         "inside_mean": np.nan, "hit@0.25": np.nan, "hit@0.5": np.nan, "hit@1.0": np.nan,
                         "delta_mae": np.nan, "delta_hit1": np.nan, "run_dir": run_dir, "status": "eval_failed"})
            continue

        # 2) Analyze (write metrics_overall.json)
        an_cmd = [
            sys.executable, "src/analyze_results.py",
            "--summary_csv", summary_csv,
            "--out_dir", analysis_dir,
            "--bins", "10", "--hist_bins", "40",
        ]
        an_rc, _ = run(an_cmd, stdout_path=os.path.join(run_dir, "analyze_stdout.txt"))
        metrics = load_metrics_json(os.path.join(analysis_dir, "metrics_overall.json"))
        if an_rc != 0 or not metrics:
            rows.append({"L_sec": L_sec, "tau": tau, "win": win, "stride": stride,
                         "mae_mean": np.nan, "mae_median": np.nan,
                         "inside_mean": np.nan, "hit@0.25": np.nan, "hit@0.5": np.nan, "hit@1.0": np.nan,
                         "delta_mae": np.nan, "delta_hit1": np.nan, "run_dir": run_dir, "status": "analyze_failed"})
            continue

        mae = float(metrics.get("mae_mean", np.nan))
        mae_med = float(metrics.get("mae_median", np.nan))
        inside = float(metrics.get("inside_mean", np.nan))
        hit025 = float(metrics.get("hit_0.25_mean", np.nan))
        hit05  = float(metrics.get("hit_0.5_mean", np.nan))
        hit10  = float(metrics.get("hit_1.0_mean", np.nan))

        # deltas (negative is better for MAE; positive is better for hits)
        d_mae  = mae - base_mae if not np.isnan(mae) and not np.isnan(base_mae) else np.nan
        d_hit1 = hit10 - base_hit1 if not np.isnan(hit10) and not np.isnan(base_hit1) else np.nan

        # Optional: light “skip-plots” heuristic if indistinguishable from baseline
        if abs(d_mae) <= args.skip_mae_tol and abs(d_hit1) <= args.skip_hit1_tol:
            # remove heavy curve images to save space
            if os.path.isdir(curves_dir):
                try:
                    for f in Path(curves_dir).glob("*.png"):
                        f.unlink()
                except Exception:
                    pass

        rows.append({
            "L_sec": L_sec, "tau": tau, "win": win, "stride": stride,
            "mae_mean": mae, "mae_median": mae_med,
            "inside_mean": inside, "hit@0.25": hit025, "hit@0.5": hit05, "hit@1.0": hit10,
            "delta_mae": d_mae, "delta_hit1": d_hit1,
            "run_dir": run_dir, "status": "ok"
        })

    # ----- Aggregate -----
    agg_csv = os.path.join(args.out_root, "aggregate.csv")
    df = pd.DataFrame(rows)
    df.to_csv(agg_csv, index=False)
    print(f"Saved aggregate to: {agg_csv}")

    # Best by MAE and by Hit@1.0
    df_ok = df[df["status"] == "ok"].copy()
    if len(df_ok):
        best_mae = df_ok.sort_values("mae_mean", ascending=True).iloc[0]
        best_hit = df_ok.sort_values("hit@1.0", ascending=False).iloc[0]

        print("\nBest-by-MAE configuration:")
        print(best_mae[["L_sec","tau","win","stride","mae_mean","mae_median","inside_mean","hit@0.25","hit@0.5","hit@1.0","delta_mae","delta_hit1","run_dir"]])

        print("\nBest-by-Hit@1.0 configuration:")
        print(best_hit[["L_sec","tau","win","stride","mae_mean","mae_median","inside_mean","hit@0.25","hit@0.5","hit@1.0","delta_mae","delta_hit1","run_dir"]])

    # ----- Heatmaps -----
    if HAVE_PLT and len(df_ok):
        pivot_mae = df_ok.pivot_table(index="L_sec", columns="tau", values="mae_mean", aggfunc="mean")
        pivot_hit = df_ok.pivot_table(index="L_sec", columns="tau", values="hit@1.0", aggfunc="mean")

        for name, pv, cmap in [("heatmap_mae", pivot_mae, "viridis"),
                               ("heatmap_hit1", pivot_hit, "magma")]:
            plt.figure(figsize=(6,4))
            im = plt.imshow(pv.values, aspect="auto", interpolation="nearest", origin="lower")
            plt.xticks(range(pv.shape[1]), [f"{c:g}" for c in pv.columns], rotation=45)
            plt.yticks(range(pv.shape[0]), [f"{r:g}" for r in pv.index])
            plt.colorbar(im)
            plt.xlabel("tau")
            plt.ylabel("L_sec")
            plt.title(name.replace("_", " "))
            plt.tight_layout()
            out_png = os.path.join(args.out_root, f"{name}.png")
            plt.savefig(out_png)
            plt.close()
            print(f"✓ {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # paths
    ap.add_argument("--vid_emb_dir", default="cache/vid_emb")
    ap.add_argument("--aud_emb_dir", default="cache/aud_emb")
    ap.add_argument("--annotations_csv", default="data/ave/ave_annotations.csv")
    ap.add_argument("--out_root", default="reports/hyper3")
    ap.add_argument("--baseline_dir", default="", help="optional: a previous run dir that has analysis/metrics_overall.json")

    # grid (focused defaults)
    ap.add_argument("--L_list", nargs="+", default=["0.25","0.5","1.0"])
    ap.add_argument("--tau_list", nargs="+", default=["0.005","0.008","0.015","0.02","0.025","0.03"])
    ap.add_argument("--win_list", nargs="+", default=["8","16"])
    ap.add_argument("--stride_list", nargs="+", default=["1","2","4"])

    # misc
    ap.add_argument("--plot_n", type=int, default=6)
    ap.add_argument("--skip_mae_tol", type=float, default=0.01, help="skip saving heavy plots if |ΔMAE| <= tol")
    ap.add_argument("--skip_hit1_tol", type=float, default=0.005, help="skip saving heavy plots if |ΔHit@1.0| <= tol")
    args = ap.parse_args()
    main(args)