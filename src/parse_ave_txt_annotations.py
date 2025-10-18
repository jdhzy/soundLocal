import csv, re
from pathlib import Path

root = Path("/restricted/projectnb/cs599dg/jd/ego4d_mc3_temporal/data/ave/raw")
ann_file = root / "Annotations.txt"

# Load split files and normalize ids (with/without .mp4)
def load_split(fname):
    p = root / fname
    if not p.exists(): return set()
    toks = set(p.read_text().split())
    norm = set()
    for t in toks:
        t = t.strip()
        if not t: continue
        if t.endswith(".mp4"): t = t[:-4]
        norm.add(t)
    return norm

splits = {
    "train": load_split("trainSet.txt"),
    "val":   load_split("valSet.txt"),
    "test":  load_split("testSet.txt"),
}

rows, bad = [], []
num_rx = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

with ann_file.open(encoding="utf-8", errors="ignore") as f:
    for ln, raw in enumerate(f, 1):
        line = raw.strip()
        if not line or line.lower().startswith(("video", "id", "header")):
            continue

        # Primary: ampersand-separated format
        if "&" in line:
            parts = [p.strip() for p in line.split("&") if p.strip() != ""]
            # Expect: [label, youtube_id, quality, start, end]
            if len(parts) >= 5:
                label = parts[0]
                vid   = parts[1]
                # quality = parts[2]  # not used
                try:
                    start_s = float(parts[-2])
                    end_s   = float(parts[-1])
                except Exception as e:
                    bad.append((ln, line, f"ampersand-float-cast-fail:{e}"))
                    continue
            else:
                bad.append((ln, line, "ampersand-too-few-fields"))
                continue
        else:
            # Fallback: first token is id, last two numbers are times, middle is label
            m = re.match(r"^\s*(\S+)\s+(.*)$", line)
            if not m:
                bad.append((ln, line, "no-id-body-match"))
                continue
            vid, body = m.group(1), m.group(2)
            nums = list(num_rx.finditer(body))
            if len(nums) < 2:
                bad.append((ln, line, "found<2-numbers"))
                continue
            n2, n1 = nums[-1], nums[-2]  # last=end, second last=start
            try:
                start_s = float(n1.group()); end_s = float(n2.group())
            except Exception as e:
                bad.append((ln, line, f"numeric-float-cast-fail:{e}"))
                continue
            label = body[:n1.start()].strip(" ,\t")

        # Determine split (check both bare id and id.mp4 forms)
        split = "unknown"
        for k, sset in splits.items():
            if vid in sset or (vid.endswith(".mp4") and vid[:-4] in sset):
                split = k
                break

        rows.append((vid, label, start_s, end_s, split))

out_csv = Path("/restricted/projectnb/cs599dg/jd/ego4d_mc3_temporal/data/ave/ave_annotations.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["video_id", "event_label", "start_s", "end_s", "split"])
    w.writerows(rows)

print(f"✅ Wrote {len(rows)} rows to {out_csv}")
if bad:
    log = Path("/restricted/projectnb/cs599dg/jd/ego4d_mc3_temporal/data/ave/annotation_parse_issues.log")
    with log.open("w", encoding="utf-8") as g:
        for ln, line, reason in bad[:300]:
            g.write(f"[line {ln}] {reason} :: {line}\n")
    print(f"⚠️ Skipped {len(bad)} lines. First 300 saved to: {log}")
