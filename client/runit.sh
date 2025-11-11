#!/bin/bash

source ../miniconda_py311/etc/profile.d/conda.sh
conda activate

ARG=$(echo "$1" | tr '[:lower:]' '[:upper:]' | tr -d '\n')


NUM_EXITS_LIST=(4 6 8 10)
DIST_LIST=("gold_ratio" "pareto" "fine" "linear")


MODEL="eenet152" ###18 or 152 #'eenet18' 'eenet34' 'eenet50' 'eenet101' 'eenet152' 'eenet20' 'eenet32' 'eenet44' 'eenet56' 'eenet110'; do
DATASET="cifar10" #cifar10, tiny-imagenet
#NUM_EXITS=4
TYPE_EXITS="pool" #bnpool, pool, plain ###plain doesn't work
EXIT_THRESHOLD=0.5
#DISTR_EXITS="pareto" #gold_ratio, pareto, fine, linear  ### linear doesn't work
BATCH=1
TRAINED_MODEL="cifar152_e10_l15_pareto_thres5.pt" # TRAINED_MODEL="model_tiny-imagenet-25_epochs_thres5.pt"
DIST_VAR="snni" #flop or snni

# Default script path for secure options



SECURE_MAIN_SCRIPT=""
SECURE_OPTION_DIR=""

if [[ -z "$ARG" ]]; then
  printf "Running plaintext EENet.\n"
  for DISTR_EXITS in "${DIST_LIST[@]}"; do
    for NUM_EXITS in "${NUM_EXITS_LIST[@]}"; do
      echo "Plaintext"$ARG" EENet with MODEL=$MODEL, NUM_EXITS=$NUM_EXITS, DISTR_EXITS=$DISTR_EXITS."
      DEVICE_MODE=cpu python plaintext/main.py --model "$MODEL" --dataset "$DATASET" --num-ee $NUM_EXITS --test-batch $BATCH --epochs 1 --testing --distribution "$DISTR_EXITS" --exit-type "$TYPE_EXITS" --exit_threshold $EXIT_THRESHOLD --load-model "$TRAINED_MODEL" --dist-method "$DIST_VAR"
    done 
  done
elif [[ "$ARG" == "C" || "$ARG" == "CLIENT" ]]; then
  SECURE_MAIN_SCRIPT="secure_option_CLIENTDETERMINED/secure_main.py"
  SECURE_OPTION_DIR="CLIENTDETERMINED"
  printf "\n\n\nRunning secure inference of SEENet, with QUOKKA OPTION CLIENT-DETERMINED.\n"
  echo "Secure option_CLIENT-DETERMINED SEENet with MODEL=s$MODEL, NUM_EXITS=$NUM_EXITS, TYPE_EXITS=$TYPE_EXITS, EXIT_THRESHOLD=$EXIT_THRESHOLD, DISTR_EXITS=$DISTR_EXITS."

elif [[ "$ARG" == "S" || "$ARG" == "SERVER" ]]; then
  SECURE_MAIN_SCRIPT="secure_option_SERVERDETERMINED/secure_main.py"
  SECURE_OPTION_DIR="SERVERDETERMINED"
  printf "\n\n\nRunning secure inference of SEENet, with QUOKKA OPTION SERVER-DETERMINED.\n"
  echo "Secure option_SERVER-DETERMINED SEENet with MODEL=s$MODEL, NUM_EXITS=$NUM_EXITS, TYPE_EXITS=$TYPE_EXITS, EXIT_THRESHOLD=$EXIT_THRESHOLD, DISTR_EXITS=$DISTR_EXITS."

elif [[ "$ARG" == "N" || "$ARG" == "NEUTRAL" ]]; then
  SECURE_MAIN_SCRIPT="secure_option_NEUTRAL/secure_main.py"
  SECURE_OPTION_DIR="NEUTRAL"
  printf "\n\n\nRunning secure inference of SEENet, with QUOKKA OPTION NEUTRAL.\n"
  echo "Secure option_NEUTRAL SEENet with MODEL=s$MODEL, NUM_EXITS=$NUM_EXITS, TYPE_EXITS=$TYPE_EXITS, EXIT_THRESHOLD=$EXIT_THRESHOLD, DISTR_EXITS=$DISTR_EXITS."

elif [[ "$ARG" == "T" || "$ARG" == "TRIAL" ]]; then
  SECURE_MAIN_SCRIPT="secure_option_CLIENTTRIAL/secure_main.py"
  SECURE_OPTION_DIR="CLIENTTRIAL"
  printf "\n\n\nRunning secure inference of SEENet, with QUOKKA OPTION CLIENT.\n"
  echo "Secure option_CLIENT SEENet with MODEL=s$MODEL, NUM_EXITS=$NUM_EXITS, TYPE_EXITS=$TYPE_EXITS, EXIT_THRESHOLD=$EXIT_THRESHOLD, DISTR_EXITS=$DISTR_EXITS."

elif [[ "$ARG" == "R" || "$ARG" == "RESNET" ]]; then
  SECURE_MAIN_SCRIPT="secure_RESNET/secure_main.py"
  SECURE_OPTION_DIR="RESNET"
  printf "\n\n\nRunning secure inference of SEENet, with QUOKKA OPTION RESNET.\n"
  echo "Secure RESNET SRESNet with MODEL=sresnet152."
  source common.sh
  source throttle.sh lan
  pids=$(lsof -t -i :29500) && [ -n "$pids" ] && echo "Killing PIDs: $pids" && echo "$pids" | xargs -r kill -9
  DEVICE_MODE=cpu python "$SECURE_MAIN_SCRIPT" --model "sresnet152" --dataset "$DATASET" --test-batch "$BATCH" --epochs 1 --testing
  exit 1

else
  printf "\n\n\nInvalid argument. Use C or CLIENT for client-determined, N or NEUTRAL for neutral, S or SERVER for server-determined options, T or TRIAL for the draft, or leave empty for plaintext inference.\n"
fi

if [[ -n "$SECURE_MAIN_SCRIPT" ]]; then
  source common.sh
  source throttle.sh lan
  pids=$(lsof -t -i :29500) && [ -n "$pids" ] && echo "Killing PIDs: $pids" && echo "$pids" | xargs -r kill -9
  DEVICE_MODE=cpu python "$SECURE_MAIN_SCRIPT" --model s"$MODEL" --dataset "$DATASET" --num-ee "$NUM_EXITS" --test-batch "$BATCH" --epochs 1 --testing --distribution "$DISTR_EXITS" --exit-type "$TYPE_EXITS" --exit_threshold "$EXIT_THRESHOLD" --load-model "$TRAINED_MODEL" --dist-method "$DIST_VAR"
fi

find . -type d -name "__pycache__" -exec rm -rf {} +