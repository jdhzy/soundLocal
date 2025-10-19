# src/hparam_sweep.py
import argparse, os, json, subprocess, sys, csv, shlex, time
from pathlib import Path

# -------------- Utility functions --------------

def parse_floats(csv_string: str):
    return [float(x.strip()) for x in csv_string.split(",") if x.strip()]

def run(cmd: str):
    """Run a shell command with visible output."""
    print(f"\033[90m$ {cmd}\033[0m")  # grey color
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise SystemExit(f"❌ Command failed ({res.returncode}): {cmd}")

def ensure_audio(lengths, wav_dir, out_dir, stride_sec, device, batch_size):
    """Generate audio embeddings for all lengths (if missing)."""
    lens_str = ",".join(str(x) for x in lengths)
    cmd = (
        f"{sys.executable} src/extract_audio_multiscale.py "
        f"--wav_dir {shlex.quote(wav_dir)} "
        f"--out_dir {shlex.quote(out_dir)} "
        f"--lengths \"{lens_str}\" "
        f"--stride_sec {stride_sec} "
        f"--device {device} "
        f"--batch_size {batch_size}"
    )
    print(f"\n🎧 Ensuring audio embeddings for lengths={lengths}...")
    run(cmd)
    print("✓ Audio embeddings ensured.\n")

# -------------- Core evaluation loop --------------

def evaluate_one(L, tau, vid_emb_dir, aud_emb_dir, ann_csv, out_root, audio_pick, plot_n):
    tag = f"L_{L}_tau_{tau}"
    out_dir = Path(out_root) / tag
    curves = out_dir / "curves"
    summary = out_dir / "summary.csv"
    analysis = out_dir / "analysis"
    curves.mkdir(parents=True, exist_ok=True)

    if summary.exists():
        print(f"↩️  Skipping {tag} (summary already exists).")
        return out_dir

    start = time.time()
    print(f"\n⏳ Running {tag}...")

    # Evaluation
    cmd_eval = (
        f"{sys.executable} src/eval_temporal_alignment.py "
        f"--vid_emb_dir {shlex.quote(str(vid_emb_dir))} "
        f"--aud_emb_dir {shlex.quote(str(aud_emb_dir))} "
        f"--annotations_csv {shlex.quote(str(ann_csv))} "
        f"--curve_dir {shlex.quote(str(curves))} "
        f"--summary_csv {shlex.quote(str(summary))} "
        f"--tau {tau} "
        f"--L_sec {L} "
        f"--audio_pick {audio_pick} "
        f"--plot_n {plot_n}"
    )
    run(cmd_eval)

    # Analysis
    cmd_an = (
        f"{sys.executable} src/analyze_results.py "
        f"--summary_csv {shlex.quote(str(summary))} "
        f"--out_dir {shlex.quote(str(analysis))} "
        f"--bins 10 --hist_bins 40"
    )
    run(cmd_an)

    elapsed = time.time() - start
    print(f"✅ Done {tag} ({elapsed/60:.1f} min)\n")
    return out_dir

# -------------- Aggregation -----------------------

def aggregate(out_root, out_csv):
    rows = []
    for metrics_path in Path(out_root).glob("L_*_tau_*/analysis/metrics_overall.json"):
        tag = metrics_path.parent.parent.name  # L_<L>_tau_<τ>
        parts = tag.split("_")
        if len(parts) != 4:
            continue
        L = float(parts[1]); tau = float(parts[3])
        with open(metrics_path, "r") as f:
            m = json.load(f)
        rows.append({
            "L_sec": L,
            "tau": tau,
            "mae_mean": m.get("mae_mean"),
            "mae_median": m.get("mae_median"),
            "inside_mean": m.get("inside_mean"),
            "hit@0.25": m.get("hit_0.25_mean"),
            "hit@0.5": m.get("hit_0.5_mean"),
            "hit@1.0": m.get("hit_1.0_mean"),
            "run_dir": str(metrics_path.parent)
        })
    if not rows:
        print("⚠️  No runs found to aggregate.")
        return
    rows.sort(key=lambda r: (r["L_sec"], r["tau"]))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"📊 Wrote aggregate summary: {out_csv}")

# -------------- Main entry -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid_emb_dir", default="cache/vid_emb")
    ap.add_argument("--aud_emb_dir", default="cache/aud_emb")
    ap.add_argument("--annotations_csv", default="data/ave/ave_annotations.csv")
    ap.add_argument("--grid_root", default="reports/grid")

    ap.add_argument("--lengths", default="0.5,1.0,2.0", help="comma list of audio crop lengths (sec)")
    ap.add_argument("--taus", default="0.03,0.05,0.07,0.10", help="comma list of temperatures")

    ap.add_argument("--audio_pick", choices=["middle"], default="middle")
    ap.add_argument("--plot_n", type=int, default=10)

    # audio extraction options
    ap.add_argument("--ensure_audio", action="store_true", help="generate missing audio embeddings")
    ap.add_argument("--wav_dir", default="data/ave/ave_wav")
    ap.add_argument("--stride_sec", type=float, default=0.25)
    ap.add_argument("--audio_device", default="cuda")
    ap.add_argument("--audio_batch_size", type=int, default=256)

    args = ap.parse_args()
    lengths = parse_floats(args.lengths)
    taus = parse_floats(args.taus)

    total_runs = len(lengths) * len(taus)
    print(f"\n🎯 Hyperparameter Sweep | total {total_runs} runs")
    print(f"Lengths: {lengths} | Taus: {taus}")
    print(f"Output root: {args.grid_root}")

    start_all = time.time()

    if args.ensure_audio:
        ensure_audio(lengths, args.wav_dir, args.aud_emb_dir, args.stride_sec, args.audio_device, args.audio_batch_size)

    # main sweep loop
    run_idx = 1
    for L in lengths:
        for tau in taus:
            print(f"\n=== [{run_idx}/{total_runs}] L={L:.2f} τ={tau:.2f} ===")
            evaluate_one(
                L=L, tau=tau,
                vid_emb_dir=args.vid_emb_dir,
                aud_emb_dir=args.aud_emb_dir,
                ann_csv=args.annotations_csv,
                out_root=args.grid_root,
                audio_pick=args.audio_pick,
                plot_n=args.plot_n
            )
            run_idx += 1

    aggregate(args.grid_root, os.path.join(args.grid_root, "aggregate.csv"))
    print(f"\n🏁 All runs complete in {(time.time() - start_all)/60:.1f} min")

if __name__ == "__main__":
    main()