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

## November 26 checkpoint

The `ef8fda3` checkpoint contained five completed development splits, nine trivia answers, and zero test predictions. Rescoring the journals against official labels on identical IDs gave the following counts.

| Task | IDs | E003 correct | Nov 26 correct | Gained | Lost |
|---|---:|---:|---:|---:|---:|
| 1 · closest integers | 60 | 60 | 60 | 0 | 0 |
| 3 · Collatz | 60 | 60 | 60 | 0 | 0 |
| 4 · concatenation | 60 | 60 | 60 | 0 | 0 |
| 5 · tweet sadness | 60 | 51 | 46 | 1 | 6 |
| 6 · same genre | 60 | 58 | 58 | 0 | 0 |
| 7 · trivia, partial | 9 | 3 | 4 | 1 | 0 |

Task 5 supplied the entire five-answer decline on the 300 completed examples. The November probe scored reasoning at 36/40 versus 30/40 for calibrated scoring, triggering reasoning for all 60 development questions. E003 used the calibrated scores for all 60. The probe and development set favored different routes.

The revised task 5 configuration uses the existing stacked label scorer directly and skips the reasoning probe. A read-only replay ran the normal task preparation and prediction path against exact request hashes. All 249 calibration responses came from the November checkpoint. All 60 development label-score responses came from E003 since the November run had only requested reasoning on those inputs. Replay recovered six answers and lost one relative to November, scoring 51/60. These development examples informed the policy choice. A fresh Qwen run and independent evaluation are needed to estimate generalization.

When continuing an older finals checkpoint, the launcher archives the superseded task 5 predictions under `prior-task5-policy`, keeps the model-response cache and schedules new predictions. Completed splits and partial task 7 progress carry forward. Task 2 and task 8 had no predictions in this checkpoint.
