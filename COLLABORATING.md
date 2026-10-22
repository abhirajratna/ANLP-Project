# Working on this repo as a team of four

Each of us runs our own Claude Code agent on our own clone. The agents talk to
each other through `tools/coord.py`. Messages, claims and status are stored as
small files on a dedicated `coord-bus` branch of this repo: one file per
message, so nothing conflicts, and the full history is in git. The rules agents
follow are in [`CLAUDE.md`](CLAUDE.md).

## One-time setup (5 minutes, each person)

1. Make sure you can push to `github.com/abhirajratna/ANLP-Project` from your
   machine (`git push` works without a password prompt).
2. Pull `main`, then pick your member name. Use lowercase, and the same name
   everywhere:
   ```bash
   git pull
   python3 tools/coord.py whoami --set <yourname>
   ```
   The first person to do this creates the `coord-bus` branch automatically.
   Run it again in every other clone or worktree you use.
3. Start Claude Code in the repo. The first time, it asks you to trust the
   project's hooks in `.claude/settings.json`; approve them.
   - **SessionStart:** shows your agent its messages, claims and the board.
   - **UserPromptSubmit:** brings in new messages, at most every 10 minutes.
   - **PreToolUse:** stops agents pushing to `main`.

   Review or turn them off with `/hooks`.

`python3` must be on your PATH (tested with 3.12; stdlib only, so no venv is needed).
On native Windows, use WSL or Git Bash.

## Day to day

You mostly don't run anything; your agent does. Useful things to tell it:

- "check the team inbox" → `coord.py inbox`
- "ask parth to rerun the floor on Dream fp16" → `coord.py send parth request ...`
- "claim the LLaDA fit-half run before submitting it" → `coord.py claim run:c1-llada-base-fit`

To see the state yourself at any time:
```bash
python3 tools/coord.py board        # everyone's status, workstream owners, all claims
python3 tools/coord.py inbox        # your open messages
python3 tools/coord.py workstreams  # areas of the repo (tools/workstreams.json) and owners
```

When the team splits the work, each person takes their area with
`python3 tools/coord.py own <workstream>`. The split in `tools/workstreams.json`
is only a starting point; change it by PR.

For a long unattended run, have the agent poll with `/loop 15m check the team inbox and act on it`.

## Things only the repo owner can do

- **Protect `main`.** On a private personal repo this needs GitHub Pro or Team.
  If it isn't available, the agent-side push guard and the rule in `CLAUDE.md`
  still apply.
  ```bash
  gh api -X PUT repos/abhirajratna/ANLP-Project/branches/main/protection --input - <<'EOF'
  {"required_status_checks": {"strict": false, "contexts": ["pytest"]},
   "enforce_admins": false,
   "required_pull_request_reviews": {"required_approving_review_count": 1},
   "restrictions": null}
  EOF
  ```
- **Do not protect `coord-bus`.** Every agent pushes to it directly.

## Troubleshooting

- **"team inbox unavailable" at session start:** run `python3 tools/coord.py board`
  in a terminal to see the real error. It is usually ssh auth or no network.
  Commands still work offline on the last synced state, but they cannot send.
- **"coord is locked by another session":** a coord command crashed mid-run.
  The lock clears itself after 2 minutes.
- **Reset your local copy of the bus.** No data is lost; it lives on the remote:
  ```bash
  git worktree remove --force .git/coord-bus
  python3 tools/coord.py board
  ```
