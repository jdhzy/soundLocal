# src/tune_temporal_alignment_expanded.py
import argparse, os, subprocess, sys, time, json, csv
from pathlib import Path
from itertools import product

def run_one(cmd, workdir):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    return p.returncode, p.stdout, dt

def main(args):
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    L_list   = [float(x) for x in args.L_list]
    tau_list = [float(x) for x in args.tau_list]
    win_list = [int(x)   for x in args.win_list]
    stride_list = [int(x) for x in args.stride_list]

    combos = list(product(L_list, tau_list, win_list, stride_list))
    total = len(combos)

    rows = []
    for i, (L, tau, win, stride) in enumerate(combos, 1):
        tag = f"L_{L}_tau_{tau}_win_{win}_stride_{stride}"
        run_dir = out_root / tag
        curves   = run_dir / "curves"
        analysis = run_dir / "analysis"
        run_dir.mkdir(parents=True, exist_ok=True)
        curves.mkdir(parents=True, exist_ok=True)
        analysis.mkdir(parents=True, exist_ok=True)

        summary_csv = run_dir / "summary.csv"

        # Skip if already done (idempotent)
        if summary_csv.exists() and not args.overwrite:
            print(f"[{i}/{total}] ✓ Skip existing {tag}")
        else:
            # 1) Evaluate
            eval_cmd = [
                sys.executable, "src/eval_temporal_alignment.py",
                "--vid_emb_dir", args.vid_emb_dir,
                "--aud_emb_dir", args.aud_emb_dir,
                "--annotations_csv", args.annotations_csv,
                "--curve_dir", str(curves),
                "--summary_csv", str(summary_csv),
                "--tau", str(tau),
                "--L_sec", str(L),
                "--audio_pick", args.audio_pick,
                "--plot_n", str(args.plot_n),
            ]
            print(f"[{i}/{total}] ▶ eval {tag}")
            rc, out, dt = run_one(eval_cmd, ".")
            # Save raw log
            (run_dir / "eval_stdout.txt").write_text(out)
            if rc != 0:
                print(f"  ✗ eval failed ({dt:.1f}s), see {run_dir/'eval_stdout.txt'}")
                continue
            print(f"  ✓ eval done in {dt:.1f}s")

            # 2) Analyze
            analyze_cmd = [
                sys.executable, "src/analyze_results.py",
                "--summary_csv", str(summary_csv),
                "--out_dir", str(analysis),
                "--bins", "10",
                "--hist_bins", "40",
            ]
            rc2, out2, dt2 = run_one(analyze_cmd, ".")
            (run_dir / "analyze_stdout.txt").write_text(out2)
            if rc2 != 0:
                print(f"  ✗ analyze failed ({dt2:.1f}s), see {run_dir/'analyze_stdout.txt'}")
            else:
                print(f"  ✓ analyze done in {dt2:.1f}s")

        # 3) Pull key metrics into memory for aggregation
        metrics_json = analysis / "metrics_overall.json"
        if metrics_json.exists():
            m = json.loads(metrics_json.read_text())
            rows.append({
                "L_sec": L,
                "tau": tau,
                "win": win,
                "stride": stride,
                "mae_mean": m.get("mae_mean"),
                "mae_median": m.get("mae_median"),
                "inside_mean": m.get("inside_mean"),
                "hit@0.25": m.get("hit_0.25_mean"),
                "hit@0.5":  m.get("hit_0.5_mean"),
                "hit@1.0":  m.get("hit_1.0_mean"),
                "run_dir": str(analysis),
            })

    # 4) Aggregate CSV
    agg_csv = out_root / "aggregate.csv"
    with agg_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "L_sec","tau","win","stride","mae_mean","mae_median","inside_mean",
            "hit@0.25","hit@0.5","hit@1.0","run_dir"
        ])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["mae_mean"] if x["mae_mean"] is not None else 1e9)):
            w.writerow(r)

    # 5) Print best by MAE
    if rows:
        best = min([r for r in rows if r["mae_mean"] is not None], key=lambda r: r["mae_mean"])
        print("\n=== Aggregated Sweep Results (saved to reports/hyper2/aggregate.csv) ===")
        for r in rows:
            print(f"L={r['L_sec']:<3}  tau={r['tau']:<4}  win={r['win']:<2}  stride={r['stride']:<2}  "
                  f"MAE={r['mae_mean']:.4f}  inside={r['inside_mean']:.4f}  "
                  f"H@0.25={r['hit@0.25']:.4f}  H@0.5={r['hit@0.5']:.4f}  H@1.0={r['hit@1.0']:.4f}")
        print("\nBest configuration by MAE:")
        for k,v in best.items():
            print(f"{k:>10}: {v}")
    else:
        print("No successful runs to aggregate.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid_emb_dir", default="cache/vid_emb")
    ap.add_argument("--aud_emb_dir", default="cache/aud_emb")
    ap.add_argument("--annotations_csv", default="data/ave/ave_annotations.csv")
    ap.add_argument("--audio_pick", default="middle")
    ap.add_argument("--plot_n", type=int, default=10)

    # sweep spaces
    ap.add_argument("--L_list", nargs="+", default=["0.25","0.5","1.0","2.0"])
    ap.add_argument("--tau_list", nargs="+", default=["0.01","0.02","0.05","0.1","0.2","0.3"])
    ap.add_argument("--win_list", nargs="+", default=["8","16","32"])
    ap.add_argument("--stride_list", nargs="+", default=["2","4","8"])

    # output root (goes to reports/hyper2)
    ap.add_argument("--out_root", default="reports/hyper2")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    main(args)