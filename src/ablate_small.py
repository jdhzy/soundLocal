# src/ablate_small.py
import argparse, os, subprocess, sys, json
from itertools import product
from time import time
from tqdm import tqdm
import pandas as pd

def run_one(args, L, tau, sigma, tau_adapt, offsets, reduce, run_root):
    tag = f"L_{L}_tau_{tau}_sg_{sigma}_Ha_{tau_adapt}_{reduce}_{offsets.replace(',','_')}"
    out_dir = os.path.join(run_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, "summary.csv")
    curve_dir   = os.path.join(out_dir, "curves")

    cmd = [
        sys.executable, "src/eval_temporal_alignment.py",
        "--vid_emb_dir", args.vid_emb_dir,
        "--aud_emb_dir", args.aud_emb_dir,
        "--annotations_csv", args.annotations_csv,
        "--summary_csv", summary_csv,
        "--curve_dir", curve_dir,
        "--L_sec", str(L),
        "--tau", str(tau),
        "--audio_pick", "multi",
        "--multi_offsets", offsets,
        "--multi_reduce", reduce,
        "--plot_n", str(args.plot_n),
        "--score_smooth_sigma", str(sigma),
        "--tau_adapt", str(tau_adapt),
        "--pred_softargmax",
        "--score_zscore",
    ]
    t0 = time()
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    took = time() - t0

    with open(os.path.join(out_dir, "stdout.txt"), "w") as f:
        f.write(r.stdout)

    ok = (r.returncode == 0 and os.path.exists(summary_csv))
    return tag, out_dir, summary_csv, ok, took

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vid_emb_dir", default="cache/vid_emb")
    ap.add_argument("--aud_emb_dir", default="cache/aud_emb")
    ap.add_argument("--annotations_csv", default="data/ave/ave_annotations.csv")
    ap.add_argument("--out_root", default="reports/ablate_small")
    ap.add_argument("--plot_n", type=int, default=0)
    # Small sweep spaces
    ap.add_argument("--L_list", type=str, default="1.0")
    ap.add_argument("--tau_list", type=str, default="0.07")
    ap.add_argument("--sigma_list", type=str, default="0.5,1.0,1.5")
    ap.add_argument("--Ha_list", type=str, default="3.0,3.5,4.0")  # entropy targets
    ap.add_argument("--offsets_list", type=str, default="0,0.25,-0.25|0,0.5,-0.5")
    ap.add_argument("--reduce_list", type=str, default="mean,max")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    Ls        = [float(x) for x in args.L_list.split(",") if x.strip()]
    taus      = [float(x) for x in args.tau_list.split(",") if x.strip()]
    sigmas    = [float(x) for x in args.sigma_list.split(",") if x.strip()]
    has       = [float(x) for x in args.Ha_list.split(",") if x.strip()]
    offsetsS  = [s for s in args.offsets_list.split("|") if s.strip()]
    reduces   = [r for r in args.reduce_list.split(",") if r.strip()]

    rows = []
    grid = list(product(Ls, taus, sigmas, has, offsetsS, reduces))
    pbar = tqdm(grid, desc="Ablate", unit="run")
    for (L, tau, sigma, Ha, offs, red) in pbar:
        pbar.set_postfix_str(f"L={L} τ={tau} σ={sigma} H*={Ha} {red} offs=[{offs}]")
        tag, run_dir, summary_csv, ok, took = run_one(args, L, tau, sigma, Ha, offs, red, args.out_root)
        row = {"tag": tag, "run_dir": run_dir, "ok": ok, "secs": round(took,1),
               "L_sec": L, "tau": tau, "sigma": sigma, "H_target": Ha, "reduce": red, "offsets": offs}
        if ok:
            try:
                df = pd.read_csv(summary_csv)
                row.update({
                    "mae_mean": float(df["mae"].mean()),
                    "inside_mean": float(df["inside"].mean()),
                    "hit@0.25": float(df["hit_0.25"].mean() if "hit_0.25" in df else df["hit@0.25"].mean()),
                    "hit@0.5":  float(df["hit_0.5"].mean()  if "hit_0.5"  in df else df["hit@0.5"].mean()),
                    "hit@1.0":  float(df["hit_1.0"].mean()  if "hit_1.0"  in df else df["hit@1.0"].mean()),
                })
            except Exception as e:
                row["ok"] = False
                row["error"] = str(e)
        rows.append(row)

    agg = pd.DataFrame(rows)
    agg.to_csv(os.path.join(args.out_root, "aggregate_small.csv"), index=False)
    print("\n=== Small Ablation Results ===")
    cols = [c for c in ["L_sec","tau","sigma","H_target","reduce","mae_mean","inside_mean","hit@0.25","hit@0.5","hit@1.0","run_dir"] if c in agg.columns]
    if cols:
        print(agg[cols].sort_values(["mae_mean"]).head(12).to_string(index=False))

if __name__ == "__main__":
    main()