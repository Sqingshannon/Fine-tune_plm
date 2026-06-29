#!/usr/bin/env bash
# Sweep RunMode.SELECTED ("position-specific", -1.0, "scores", "full") over
# shots [48, 64, 96, 144, 192, 240], all 4 quarters, seeds 1-5.
#
# Each quarter is its own independent stream that loops through ALL 6 shots
# back-to-back (main pass, then a "rerun missing" pass) with NO
# synchronization barrier against the other quarters. A quarter that
# finishes early just keeps going to its next shot/pass immediately instead
# of waiting on the others.
#
# GPU pinning (fixed for the whole run):
#   q1 -> GPUs 0,1,2     q2 -> GPUs 0,1,2   (ALWAYS share, run concurrently)
#   q3 -> GPUs 3,4,5     q4 -> GPUs 3,4,5   (ALWAYS share, run concurrently)
#   GPUs 6,7 are unused.
#
# RunMode.SELECTED already sets skip_done=True, so the rerun pass instantly
# skips anything already completed and only retrains genuinely missing/
# failed (dataset, shot, seed, combo) entries.

set -uo pipefail

# Run from the repo root regardless of where this script is invoked from --
# all paths in psifit/ (config/, data_rerun_fixed/, checkpoint_*/, predicted_*/)
# are relative to cwd.
cd "$(dirname "$(readlink -f "$0")")"

SHOTS=(48 64 96 144 192 240)
SEEDS=(1 2 3 4 5)
LOG_DIR="logs/shot_sweep"
mkdir -p "$LOG_DIR"

run_job() {
    local quarter=$1
    local gpus=$2
    local shot=$3
    local pass=$4
    local logfile="$LOG_DIR/shot${shot}_q${quarter}.log"
    echo "  -> [GPUs $gpus] pass=$pass shot=$shot quarter=$quarter (log: $logfile)"
    CUDA_VISIBLE_DEVICES="$gpus" python -m psifit.run_entry \
        --quarter "$quarter" \
        --mode selected \
        --shot "$shot" \
        --model_seeds "${SEEDS[@]}" \
        > "$logfile" 2>&1
    echo "  <- [GPUs $gpus] pass=$pass shot=$shot quarter=$quarter finished (exit $?)"
}

run_quarter_stream() {
    local quarter=$1
    local gpus=$2
    local pass shot
    for pass in 1 2; do
        for shot in "${SHOTS[@]}"; do
            run_job "$quarter" "$gpus" "$shot" "$pass"
        done
    done
    echo "=== quarter=$quarter (GPUs $gpus) fully done: both passes, all shots ==="
}

echo "##### Launching 4 independent quarter streams (no cross-quarter wait) #####"

run_quarter_stream 1 "0,1,2" & pid1=$!
run_quarter_stream 2 "0,1,2" & pid2=$!
run_quarter_stream 3 "3,4,5" & pid3=$!
run_quarter_stream 4 "3,4,5" & pid4=$!

wait "$pid1" "$pid2" "$pid3" "$pid4"

echo "##### Shot sweep complete. Check $LOG_DIR/ for per-quarter logs. #####"
