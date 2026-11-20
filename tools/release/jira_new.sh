#!/usr/bin/env bash
# Create one Jira issue under the release epic and record its key in docs/release-1.0/tracking.json.
# Usage: tools/release/jira_new.sh <slug> <Type> "<Summary>" <description-file> [extra,labels]
# Refuses if the slug is already recorded (Jira issues cannot be deleted by this tooling).
set -euo pipefail
SKILLS="${DCRI_SKILLS_DIR:-$HOME/PROJECTS/devops/dcri-skills-setup/plugins/dcri-global/skills}"
JC="$SKILLS/jira-confluence/scripts/jc"
TRACK="${TRACK:-docs/release-1.0/tracking.json}"
EPIC_TRACK="docs/release-1.0/tracking.json"
slug="$1"; type="$2"; summary="$3"; desc="$4"; labels="data-analyzer,release-1.0${5:+,$5}"
[ -f "$TRACK" ] || echo '{"epic": null, "issues": {}, "pages": {}}' > "$TRACK"
python3 - "$TRACK" "$slug" <<'PY'
import json,sys
t=json.load(open(sys.argv[1]))
if sys.argv[2] in t["issues"]: sys.exit(f"already recorded: {sys.argv[2]} = {t['issues'][sys.argv[2]]['key']}")
PY
epic=$(python3 -c "import json;print(json.load(open('$EPIC_TRACK'))['epic'] or '')")
out=$(bash "$JC" jira create --project ADP --type "$type" --summary "$summary" --description-file "$desc" \
      --labels "$labels" --assignee me ${epic:+--parent "$epic"})
python3 - "$TRACK" "$slug" "$type" "$summary" "$out" <<'PY'
import json,sys
p,slug,typ,summary,out=sys.argv[1:]
r=json.loads(out); t=json.load(open(p))
t["issues"][slug]={"key":r["key"],"id":r["id"],"type":typ,"summary":summary}
json.dump(t,open(p,"w"),indent=1); print(r["key"], r["url"])
PY
