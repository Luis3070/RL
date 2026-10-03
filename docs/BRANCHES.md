# Branch registry and consolidation record

Computed on 2026-10-03 from the local clone, without fetching (remote-tracking refs may be stale).
Ahead/behind is `git rev-list --left-right --count origin/main...<ref>` (left = commits only on `origin/main`, right = commits only on the branch).
Nothing was pushed, tagged or deleted. `origin` belongs to a third-party GitHub account; deleting or tagging remote branches is a decision for the maintainers and requires their authorization. All entries in the last column are recommendations only.

"Contained in" means `git merge-base --is-ancestor <tip> <ref>` returned true.

## Remote heads (13)

| Branch | Tip | Last commit | Behind / ahead of `origin/main` | Outcome | Recommendation |
|---|---|---|---|---|---|
| `main` | f811ef5 | 2026-04-15 | 0 / 0 | Base reference; untouched | keep |
| `rl-v4.2-multiagent-gk` | 66399aa | 2026-07-08 | 0 / 86 | Integration tip; contains the four fully absorbed branches below | keep |
| `rl-simrobot3d` | 4c8a43d | 2026-06-13 | 0 / 80 | Absorbed in full by fast-forward (contained in `rl-v4.2-multiagent-gk`) | tag candidate |
| `pybh-clean` | 9db4f46 | 2026-04-28 | 0 / 9 | Absorbed in full by fast-forward (contained in `rl-v4.2-multiagent-gk`) | tag candidate |
| `rlsimrobot` | 0e97059 | 2026-04-22 | 0 / 1 | Absorbed in full by fast-forward (contained in `rl-v4.2-multiagent-gk`) | tag candidate |
| `TKREADY-0.1-TKSUCC-0.9-TIME-56MS` | 4110710 | 2026-04-11 | 3 / 0 | Absorbed in full (contained in `rl-v4.2-multiagent-gk`; no commits beyond `origin/main` history) | tag candidate |
| `rl-v3a-multiagent-defender` | 2a85d0f | 2026-06-19 | 0 / 85 | Merged (commit `ea1a71c`); contained in the consolidation branch `chore/organizar-repo-paper`, not in `rl-v4.2-multiagent-gk` | tag candidate |
| `obs-26-final` | fbe1e98 | 2026-06-12 | 0 / 65 | Merged (3 commits; docs 28-30 renumbered to 40-42); contained in `chore/organizar-repo-paper` | tag candidate |
| `Defender-v7` | ce43438 | 2026-06-16 | 0 / 83 | Partial: only `docs/35-defender-v7-block-pass-gate-fixes.md` kept; its block-rule code is superseded by the v4.2/v5 design | archive candidate |
| `pipeprocess` | ff1a6fa | 2026-04-14 | 2 / 5 | Partial: legacy April tools absorbed, then removed as outdated for the paper | archive candidate |
| `Bridge-Working-Linux` | d6e470f | 2026-04-15 | 2 / 6 | Partial: legacy April tools absorbed, then removed as outdated for the paper | archive candidate |
| `telemetry` | c31e8e2 | 2026-04-15 | 2 / 6 | Partial: legacy April tools absorbed, then removed as outdated for the paper | archive candidate |
| `test-multiagent` | 03afe59 | 2026-06-12 | 0 / 79 | Skipped (its tool is already present) | archive candidate |

## Local-only branches

| Branch | Tip | Last commit | Behind / ahead of `origin/main` | Outcome | Recommendation |
|---|---|---|---|---|---|
| `chore/organizar-repo-paper` | e9e7b98 at the time of writing (moves as documents are added) | 2026-10-03 | 0 / 104 | Consolidation branch for the paper repository; 18 commits ahead of `origin/rl-v4.2-multiagent-gk` | unpushed-at-risk |
| `rl-v3b-goalkeeper` | 66177ca | 2026-06-19 | 0 / 85 | Partial: new goalkeeper scripts kept, then pruned. Not contained in any remote branch (`git branch -r --contains` returns nothing) | unpushed-at-risk |
| `docs/organize-paper-repo` | f811ef5 | 2026-04-15 | 0 / 0 | Same commit as `main`; no unique content | archive candidate (local housekeeping) |

Local `main` and `rl-v4.2-multiagent-gk`, `rl-v3a-multiagent-defender` track their `origin/*` counterparts at the same tips.

## Commands used

```bash
git branch -a -vv
for b in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin refs/heads); do
  echo "$b $(git rev-parse --short $b) $(git log -1 --format='%ad|%an' --date=short $b) $(git rev-list --left-right --count origin/main...$b)"
done
git merge-base --is-ancestor <tip> origin/rl-v4.2-multiagent-gk     # containment checks
git merge-base --is-ancestor <tip> chore/organizar-repo-paper
git branch -r --contains rl-v3b-goalkeeper
git log --oneline origin/rl-v4.2-multiagent-gk..chore/organizar-repo-paper
```

## Notes

- The C++ repository `SabanaHerons2026` has its own branches (for example `rl-simrobot3d` and `hsl2026-ppo-team-v4`); this registry covers only the Python repository.
- Tagging before any deletion would keep the history reachable; this is only a suggestion.
