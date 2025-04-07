#!/bin/bash
# Run it in MeG-for-Knowledge-Editing directory

# --- config ---
TOTAL_DATA=1024  # num of data used
NUM_JOBS=2      # total num of processes
GPU_SELECTION_RANGE="3 4"
BASE_OUTPUT_DIR="output"
SCRIPT_TO_RUN="evaluate/run_all_evaluation_Parallel.py"
PYTHON_EXECUTABLE="python"
# Choose which model and dataset to use
MODEL="gptj"
DATA_TYPE="zsre"


JOB_COUNTER=0

# --- calculate the amount of basic data and the remainder for each task ---
BASE_DATA_PER_JOB=$(( TOTAL_DATA / NUM_JOBS ))
REMAINDER=$(( TOTAL_DATA % NUM_JOBS ))

# --- create a top-level timestamp directory ---
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
TOP_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "$TOP_OUTPUT_DIR"

run_job() {
  local job_id=$1
  local gpu_id=$2
  local top_output_dir=$3

  # ---  Calculate the amount of data processed by the current job (considering uneven data distribution) ---
  local data_per_job_for_this_job
  if [ $job_id -lt $NUM_JOBS ]; then
    # The first (NUM_JOBS-1) job allocates BASE_DATA_PER_JOB data to each job
    data_per_job_for_this_job=$BASE_DATA_PER_JOB
  else
    # assign all the remaining data to the last job
    data_per_job_for_this_job=$((BASE_DATA_PER_JOB + REMAINDER))
  fi

  local start_index=$(( (job_id - 1) * BASE_DATA_PER_JOB ))
  local end_index=$(( start_index + data_per_job_for_this_job))

  # make sure the end index does not exceed TOTAL DATA
  if [ $end_index -gt $TOTAL_DATA ]; then
    end_index=$TOTAL_DATA
  fi

  # build output directories and log files, with each process having its own log files
  local log_file="${top_output_dir}/log_${job_id}.txt"
  touch "$log_file"

  echo "Running job $job_id on GPU $gpu_id (data range: $start_index-$end_index)"
  $PYTHON_EXECUTABLE $SCRIPT_TO_RUN \
    --start_index "$start_index" \
    --end_index "$end_index" \
    --data_range "$TOTAL_DATA" \
    --is_parallel True \
    --time_stamp "$TIMESTAMP" \
    --gpu_id "$gpu_id" \
    --model_para_type "$MODEL" \
    --data_type "$DATA_TYPE" \
     > "$log_file" 2>&1 & 

  echo "Job $job_id (PID $!) started on GPU $gpu_id"
  sleep 10  # wait 10 seconds to prevent folder creation conflicts during initialization
}

timer_start=$(date "+%Y-%m-%d %H:%M:%S")
echo "start time：$timer_start"

while [ $JOB_COUNTER -lt $NUM_JOBS ]; do
  # --- iterate over the specified GPU range ---
  for gpu_id in $GPU_SELECTION_RANGE; do
    gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk -v line=$((gpu_id+1)) 'NR==line {print $1}')
    echo "GPU $gpu_id free memory: $gpu_free"

    # If GPU memory > 10GB, start a job
    if [ "$gpu_free" -gt 10000 ]; then
      JOB_COUNTER=$((JOB_COUNTER + 1))
      run_job "$JOB_COUNTER" "$gpu_id" "$TOP_OUTPUT_DIR"
      echo "One job started on GPU $gpu_id."

      # all jobs started
      if [ $JOB_COUNTER -ge $NUM_JOBS ]; then
        break
      fi
    fi
  done

  # , exit the main loop if all jobs are started
  if [ $JOB_COUNTER -ge $NUM_JOBS ]; then
    break
  fi

  # When there are not enough GPUs for all programs, wait a while and check again
  sleep 30
done

wait

timer_end=$(date "+%Y-%m-%d %H:%M:%S")
echo "end time：$timer_end"

start_seconds=$(date --date="$timer_start" +%s);
end_seconds=$(date --date="$timer_end" +%s);

echo "Total execution time: "$((end_seconds-start_seconds))" seconds"
echo "Total execution time: "$((end_seconds-start_seconds))" seconds" > "${TOP_OUTPUT_DIR}/total_time.txt"

# --- merge results ---
$PYTHON_EXECUTABLE merge_results.py \
  --time_stamp "$TIMESTAMP" \
  --data_range "$TOTAL_DATA"

echo "All jobs completed."