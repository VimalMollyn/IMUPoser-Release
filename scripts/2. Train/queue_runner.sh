#!/usr/bin/env bash
# File-driven experiment queue for one GPU. Each line of the queue file is a run_newdata.sh spec
# ("<tag>|<arm>|<seed>[|ENV=.. ENV=..]"); lines starting with # are ignored. The runner takes the FIRST
# line not yet in <queue>.done / <queue>.running, runs it (base -> FT -> eval), appends it to .done, repeats.
# Edit/reorder/append the queue file at any time; an empty queue just polls every minute.
#   queue_runner.sh <gpu> <queue_file>
set -u
cd "$(dirname "$0")"
GPU="${1:?gpu}"; Q="${2:?queue file}"
touch "$Q.done" "$Q.running"
log(){ echo "[queue gpu$GPU $(date -Is)] $*"; }
while true; do
  spec=""
  while IFS= read -r line; do
    [ -z "$line" ] && continue; case "$line" in \#*) continue;; esac
    tag="${line%%|*}"
    grep -qx "$tag" "$Q.done" && continue
    grep -qx "$tag" "$Q.running" && continue
    spec="$line"; break
  done < "$Q"
  if [ -z "$spec" ]; then sleep 60; continue; fi
  tag="${spec%%|*}"
  echo "$tag" >> "$Q.running"
  log "START $spec"
  bash run_newdata.sh "$GPU" "$spec"
  log "END $tag"
  grep -vx "$tag" "$Q.running" > "$Q.running.tmp"; mv "$Q.running.tmp" "$Q.running"
  echo "$tag" >> "$Q.done"
done
