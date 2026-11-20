Review the security of the code before release, using two independent reviews, and track every confirmed finding to a fix or a recorded decision.

# Acceptance criteria
1. Two reviews are done independently (Claude Code and Codex), each without sight of the other's findings. The Claude Code review is finished and its report fingerprinted before the Codex report is opened.
2. Every confirmed finding has its own Jira issue under this epic, with severity and the decision taken.
3. High severity findings are fixed and re-tested before release, or the release is held.
4. Hardening items that are not vulnerabilities are listed separately so they are not lost.

Details of unfixed findings are kept in Jira only. The code repository is public.
