# Running the finals across Kaggle sessions

On November 25, 2026 Kaggle reported that the finals notebook exceeded its maximum execution duration (exit 137). The last completed task and the amount of recoverable output were unknown. The original launcher interleaved each development batch with its full test batch. In E003, counting alone took 7,719 seconds for preparation and 60 development answers. A full finals run requires another 500 test answers on that task.

Use two T4 GPUs and enable Internet. If the stopped notebook saved a `cairn-finals` output folder, attach that notebook output to the next session. The launcher also resumes directly from an existing `/kaggle/working/cairn-finals`. It recognizes the run identity from commit `84ce0cc`. Each fresh session needs the most recent saved output to continue.

```bash
%%bash
set -euo pipefail
repo=$(mktemp -d /tmp/cairn-finals.XXXXXX)
git clone -q --depth 1 -b codex/kaggle-e004-comparison https://github.com/madhesh60/ICL.git "$repo"
CAIRN_SESSION_MINUTES=180 bash "$repo/scripts/kaggle_e004.sh" finals
```

The budget runs from launcher startup, including dependency and model setup. Choose a budget shorter than the time left in the Kaggle session, allowing additional time for shutdown and notebook output saving. At `CHECKPOINT READY`, save the output and attach it as a notebook input for the next run. The response caches and journals are both useful. If Kaggle discarded the stopped session's output, the next run starts fresh.

Every five minutes the launcher prints test IDs saved and available development scores. It checks the 60 development IDs and recomputes correct counts before skipping finished work. It validates completed test files against all official IDs. The planner runs the development batches first and leaves the full query set available to the context builder when resuming a partially finished task. This preserves the retrieval inputs and calibration settings. A clean restart can still need model loading, context construction and calibration.

The submission contains 500 test predictions for each of tasks 1–7 and 166 for task 8. `submission.zip` is produced after the eight files pass validation. Each completed GPU kernel replay is saved separately, so the final numerical checks can span sessions too. `FINALS COMPLETE` marks the end of the replay. Local development accuracy and kernel consistency are reported separately. Only the organizer can score the hidden test labels.
