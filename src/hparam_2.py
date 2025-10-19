#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter sweep for temporal alignment with clean progress + preflight checks.

Creates structure:
  reports/<out_root>/<RUN_NAME>/
    summary.csv
    curves/...
    analysis/
      metrics_overall.json
      hist_mae.png (if MAE available)
      calibration.png (if confidence available)
  reports/<out_root>/aggregate.csv
"""

import os, sys, argparse, json, glob, shlex, subprocess
from pathlib import Path
from itertools import product
from tqdm import tqdm
import pandas as pd

# ----------------------------- helpers -----------------------------

def shell(cmd, env=None, cwd=None, quiet=True, log_path=None):
    """Run a shell command. If quiet, capture output and optionally write to log."""
    if not quiet:
        print(f"$ {cmd}")
    proc = subprocess.run(
        cmd, shell=True, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            f.write(proc.stdout)
    return proc.returncode, proc.stdout

def has_audio_npzs(aud_emb_dir: str, L_sec: float) -> bool:
    ms = int(round(L_sec * 1000))
    pattern = str(Path(aud_emb_dir) / f"*__L{ms}ms.npz")
    return len(glob.glob(pattern)) > 0

def count_npz(dir_path: str) -> int:
    return len(glob.glob(str(Path(dir_path) / "*.npz")))

def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid_emb_dir", default="cache/vid_emb")
    ap.add_argument("--aud_emb_dir", default="cache/aud_emb")
    ap.add_argument("--annotations_csv", default="data/ave/ave_annotations.csv")

    ap.add_argument("--out_root", default="reports/hyper2")
    ap.add_argument("--plot_n", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="suppress per-run stdout (kept in log files)")

    # grids
    ap.add_argument("--L_list", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--tau_list", nargs="+", type=float, default=[0.03, 0.05, 0.07, 0.10])
    ap.add_argument("--win_list", nargs="+", type=int,   default=[16])
    ap.add_argument("--stride_list", nargs="+", type=int, default=[4])

    args = ap.parse_args()

    vid_emb_dir   = Path(args.vid_emb_dir)
    aud_emb_dir   = Path(args.aud_emb_dir)
    annotations   = Path(args.annotations_csv)
    out_root      = Path(args.out_root)

    # ---------- preflight checks ----------
    issues = []
    if count_npz(vid_emb_dir) == 0:
        issues.append(f"No video embeddings in {vid_emb_dir}")
    if not annotations.exists():
        issues.append(f"Annotations missing: {annotations}")
    missing_L = [L for L in args.L_list if not has_audio_npzs(aud_emb_dir, L)]
    if missing_L:
        issues.append("Missing audio crops for lengths: " + ", ".join(map(str, missing_L)))

    preflight_path = out_root / "_preflight.json"
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(preflight_path, {
        "vid_emb_dir": str(vid_emb_dir),
        "aud_emb_dir": str(aud_emb_dir),
        "annotations_csv": str(annotations),
        "counts": {
            "vid_npz": count_npz(vid_emb_dir),
            "aud_npz_total": count_npz(aud_emb_dir)
        },
        "missing_L": missing_L
    })

    if issues:
        print("✗ Preflight failed:")
        for s in issues:
            print("  -", s)
        print(f"\nDetails written to {preflight_path}")
        sys.exit(2)

    # ---------- build grid ----------
    grid = []
    for (L, tau, win, stride) in product(args.L_list, args.tau_list, args.win_list, args.stride_list):
        name = f"L_{L}_tau_{tau}_win_{win}_stride_{stride}".replace(".", "_")
        run_dir = out_root / name
        curve_dir   = run_dir / "curves"
        summary_csv = run_dir / "summary.csv"
        analysis_dir= run_dir / "analysis"
        grid.append({
            "name": name,
            "L_sec": L,
            "tau": tau,
            "win": win,
            "stride": stride,
            "run_dir": run_dir,
            "curve_dir": curve_dir,
            "summary_csv": summary_csv,
            "analysis_dir": analysis_dir
        })

    # ---------- sweep with progress bar ----------
    # Use a non-interactive backend for matplotlib inside analyze
    env = os.environ.copy()
    env["MPLBACKEND"] = env.get("MPLBACKEND", "Agg")

    pbar = tqdm(grid, desc="Hyperparam sweep", unit="cfg")

    results = []
    for cfg in pbar:
        pbar.set_postfix_str(f"L={cfg['L_sec']}, τ={cfg['tau']}, W={cfg['win']}, S={cfg['stride']}")

        cfg["run_dir"].mkdir(parents=True, exist_ok=True)
        cfg_ok = True

        # --- EVAL ---
        if not cfg["summary_csv"].exists() or args.overwrite:
            eval_cmd = (
                "python src/eval_temporal_alignment.py "
                f"--vid_emb_dir {shlex.quote(str(vid_emb_dir))} "
                f"--aud_emb_dir {shlex.quote(str(aud_emb_dir))} "
                f"--annotations_csv {shlex.quote(str(annotations))} "
                f"--curve_dir {shlex.quote(str(cfg['curve_dir']))} "
                f"--summary_csv {shlex.quote(str(cfg['summary_csv']))} "
                f"--tau {cfg['tau']} --L_sec {cfg['L_sec']} "
                f"--audio_pick middle --plot_n {args.plot_n} "
                f"--hit_deltas 0.25 0.5 1.0"
            )
            rc, out = shell(eval_cmd, env=env, quiet=args.quiet,
                            log_path=str(cfg["run_dir"] / "eval_stdout.txt"))
            if rc != 0:
                cfg_ok = False
        else:
            # already evaluated
            pass

        # --- ANALYZE ---
        if cfg_ok and cfg["summary_csv"].exists():
            analyze_cmd = (
                "python src/analyze_results.py "
                f"--summary_csv {shlex.quote(str(cfg['summary_csv']))} "
                f"--out_dir {shlex.quote(str(cfg['analysis_dir']))} "
                f"--bins 10 --hist_bins 40"
            )
            rc, out = shell(analyze_cmd, env=env, quiet=args.quiet,
                            log_path=str(cfg["run_dir"] / "analyze_stdout.txt"))
            if rc != 0:
                cfg_ok = False

        # --- harvest metrics_overall.json (if present) ---
        metrics_path = cfg["analysis_dir"] / "metrics_overall.json"
        if cfg_ok and metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metr = json.load(f)
                results.append({
                    "L_sec": cfg["L_sec"],
                    "tau": cfg["tau"],
                    "win": cfg["win"],
                    "stride": cfg["stride"],
                    "mae_mean": metr.get("mae_mean"),
                    "mae_median": metr.get("mae_median"),
                    "inside_mean": metr.get("inside_mean"),
                    "hit@0.25": metr.get("hit_0.25_mean"),
                    "hit@0.5": metr.get("hit_0.5_mean"),
                    "hit@1.0": metr.get("hit_1.0_mean"),
                    "run_dir": str(cfg["analysis_dir"])
                })
            except Exception as e:
                # keep going; logs are in analyze_stdout.txt
                pass

    # ---------- aggregate ----------
    agg_path = out_root / "aggregate.csv"
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(
            by=["mae_mean", "inside_mean", "hit@1.0"],
            ascending=[True, False, False]
        )
        df.to_csv(agg_path, index=False)
        print("\n=== Aggregated Sweep Results ===")
        show_cols = [c for c in ["L_sec","tau","win","stride","mae_mean","inside_mean","hit@0.25","hit@0.5","hit@1.0"] if c in df.columns]
        print(df[show_cols].to_string(index=False))
        print("\nBest configuration:")
        print(df.iloc[0].to_string())
        print(f"\nSaved aggregate to: {agg_path}")
    else:
        print("\n(no successful runs to aggregate)")
        print("Check per-run logs under:", out_root)

if __name__ == "__main__":
    main()