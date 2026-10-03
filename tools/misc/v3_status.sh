#!/usr/bin/env bash
# One-shot status for the v3 regen. Run: bash tools/misc/v3_status.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../scripts/env.sh"
cd "$RL_REPO_DIR"
LOG="$(cat /tmp/v3_regen_log.txt 2>/dev/null)"
RUN=$(pgrep -f "run_v3_attack|teacher_multiagent_dataset" | grep -v grep | wc -l)
echo "============ v3 regen status ============"
if [ "$RUN" -gt 0 ]; then
  echo "STATUS : RUNNING ($RUN procs)"
elif grep -q "regen done" "$LOG" 2>/dev/null; then
  echo "STATUS : DONE ✅  → next:  bash tools/run/run_v3_attack.sh merge"
else
  echo "STATUS : STOPPED (not 'done') — inspect log tail below"
fi
echo "chunks : $(ls -d runs/teacher_multiagent_dataset/v3_chunks/*/ 2>/dev/null | wc -l) / 9"
echo "sims   : SimRobot=$(pgrep -f SimRobot|wc -l)"
echo "--- contrastive keep counts (per stage-7 chunk; want kept>0) ---"
~/anaconda3/envs/RLNAO/bin/python - <<'PY'
import json, glob, os
chunks=sorted(glob.glob("runs/teacher_multiagent_dataset/v3_chunks/dataset_stage7_*"))
if not chunks: print("  (no stage-7 chunk yet — still on stage 6)")
for d in chunks:
    p=os.path.join(d,"summary.json")
    p=p if os.path.exists(p) else p.replace("summary.json","summary.partial.json")
    if not os.path.exists(p): continue
    rows=[s for s in json.load(open(p)).get("scenarios",[]) if "giveandgo" in s["name"] or "support_" in s["name"]]
    tag=os.path.basename(d)+("" if os.path.exists(os.path.join(d,"summary.json")) else " [partial]")
    print(f"  {tag}")
    for s in rows: print(f"    {s['name']:24s} kept={s['kept_episodes']}/{s['attempted_episodes']}")
PY
echo "--- log tail ---"; tail -5 "$LOG" 2>/dev/null
