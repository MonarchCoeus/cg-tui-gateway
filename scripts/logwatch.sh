#!/usr/bin/env bash
# Live view of the CG gateway's per-request log, polled from /v1/logs.
# Opened by the TUI's 'l' key; also runnable by hand:
#   bash ~/Projects/CG/scripts/logwatch.sh  (set CG_LOG_URL to override)
URL="${CG_LOG_URL:-http://127.0.0.1:20185/v1/logs}"
LATEST=30
while :; do
  clear
  echo "CG live log — $(date '+%H:%M:%S')   (ctrl-c to close)"
  echo "--------------------------------------------------------"
  if ! curl -s --max-time 3 "${URL}?n=${LATEST}" 2>/dev/null | python3 -c '
import json, sys, time
try:
    entries = json.load(sys.stdin)["entries"]
except Exception:
    print("(gateway unreachable — no log endpoint yet?)")
    sys.exit(0)
for e in reversed(entries):
    print("%s  %-33s key=%-6s status=%-3s %8sms" % (
        time.strftime("%H:%M:%S", time.localtime(e["t"])),
        e["model"], e["key"], e["status"], e["ms"]))
'; then
    echo "(poll failed)"
  fi
  sleep 1
done
