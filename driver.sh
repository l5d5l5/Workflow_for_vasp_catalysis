#!/bin/bash
set -euo pipefail

ROOT=/data2/home/luodh/high-calc-2
cd "$ROOT"

PYTHON=/data2/home/luodh/anaconda3/envs/workflow/bin/python
PARAMS=$ROOT/params.yaml
LOG=$ROOT/runs/driver.log

mkdir -p "$ROOT/runs"

# 每轮最多提交多少个新任务（按队列容量调）
LIMIT=50
# 轮询间隔（秒）
SLEEP=7200

while true; do
  echo "===== $(date) submit-all begin =====" >> "$LOG"
  "$PYTHON" hook.py --params "$PARAMS" submit-all --limit "$LIMIT" >> "$LOG" 2>&1 || true
  echo "===== $(date) submit-all end =====" >> "$LOG"
  sleep "$SLEEP"
done
