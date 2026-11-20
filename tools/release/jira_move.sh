#!/usr/bin/env bash
# Move a Jira issue through one or more transitions and log the time, so the
# issue history (and docs/release-1.0/timeline.txt) can be read as a timeline.
# Usage: tools/release/jira_move.sh KEY "Transition name" ["Next transition" ...]
set -euo pipefail
SKILLS="${DCRI_SKILLS_DIR:-$HOME/PROJECTS/devops/dcri-skills-setup/plugins/dcri-global/skills}"
JC="$SKILLS/jira-confluence/scripts/jc"
key="$1"; shift
for t in "$@"; do
  bash "$JC" jira transition "$key" "$t" >/dev/null
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') | $key | -> $t" >> docs/release-1.0/timeline.txt
done
bash "$JC" jira issue "$key" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['key'], d['status'])"
