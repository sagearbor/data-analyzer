# Usage Notes: data-analyzer-api skill

## Where this works today

**Claude Code (this skill): works now.** Any Claude Code session running in
or against this repo can load `SKILL.md` and call the deployed API directly
via `curl` from the Bash tool, as long as the operator is on Duke VPN and the
repo's `.env` has `DATA_ANALYZER_API_KEY` set. No additional wiring is
required — the skill is self-contained shell instructions.

## Where this is parked

**Claude Desktop (MCP-server path): not yet available.** The natural next
step would be to expose this API as an MCP server/connector so it's callable
directly from Claude Desktop (or claude.ai) without a terminal. That work is
**intentionally on hold** pending answers from Anthropic's account team on:

- **ZDR (Zero Data Retention)** status/requirements for this deployment and
  whether an MCP connector to an internal, VPN-gated API is compatible with
  Duke's data handling posture.
- Confirmation of the appropriate connector pattern (remote MCP server vs.
  local/desktop extension) given the API is not internet-reachable.

Until that's resolved, do not build or wire up an MCP server for this API.
The Claude Code + curl approach in `SKILL.md` is the supported path.

## Standing constraint either way

The underlying `data-analyzer-api` service is **VPN-only** (Duke network). No
integration path — Claude Code today, or a future Claude Desktop/MCP path —
changes that. Any caller, human or agent, must be on VPN to reach it. This is
a network-level restriction on the Azure Container Apps ingress, not
something client-side tooling can work around.
