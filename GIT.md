# Git workflow (School Knowledge Base)

Quick reference for adding, removing, and publishing code on Windows (PowerShell).

## See what changed

```powershell
cd d:\schoolknowledgebase
git status
git diff                    # unstaged changes
git diff --staged           # staged changes (ready to commit)
```

## Add code (include in next commit)

```powershell
# One file
git add app/services/exam_generator.py

# Whole folder
git add app/prompts/

# All modified tracked files
git add -u

# All changes including new files (respects .gitignore)
git add .
```

**Not uploaded** (see `.gitignore`): `.venv/`, `logs/`, `.env`.

## Remove code from Git

| Goal | Command |
|------|---------|
| Stop tracking a file but **keep it on disk** | `git rm --cached path\to\file` |
| Stop tracking a folder but **keep it on disk** | `git rm -r --cached logs` |
| Delete from repo **and** delete the file locally | `git rm path\to\file` |
| Undo edits to a file (before commit) | `git restore path\to\file` |
| Unstage a file (keep your edits) | `git restore --staged path\to\file` |

Example — stop tracking logs after adding them to `.gitignore`:

```powershell
git rm -r --cached logs
git add .gitignore
git commit -m "Stop tracking logs and ignore .venv"
```

## Commit and push

```powershell
git commit -m "Short message: what you changed and why"
git push

# First push on a new branch:
git push -u origin your-branch-name
```

## Typical day: edit → add → commit → push

```powershell
git status
git add app/services/exam_generator.py app/schemas/exam.py
git commit -m "Fix exam generation for missing chapters"
git push
```

## Undo mistakes

```powershell
# Discard local edits to one file (not committed yet)
git restore app/services/exam_generator.py

# Undo last commit, keep files changed (fix message or add more files)
git reset --soft HEAD~1

# See history
git log --oneline -10
```

## Branches (optional)

```powershell
git checkout -b feature/my-change
# ... work, commit ...
git push -u origin feature/my-change
```

## What `.gitignore` does

| Path | Effect |
|------|--------|
| `.venv/` | Virtual environment never uploaded |
| `logs/` | Log files never uploaded |
| `.env` | Secrets never uploaded |

If something was **already committed** before it was ignored, run `git rm --cached ...` once, then commit. Ignore rules alone do not remove old tracked files.

## Quick cheat sheet

- **Add to next commit:** `git add <path>`
- **Remove from repo only:** `git rm --cached <path>`
- **Remove from repo and disk:** `git rm <path>`
- **Drop local edits (not committed):** `git restore <path>`
- **Unstage:** `git restore --staged <path>`
