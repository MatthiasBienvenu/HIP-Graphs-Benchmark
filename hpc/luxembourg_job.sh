#!/bin/bash

#OAR -q default
#OAR -l host=1/gpu=1,walltime=3:00:00
#OAR -p mi210
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err

module load python/3.10.8_gcc-10.4.0

source ~/.venv-mi210/bin/activate

cd ~/HIP-Graphs-Benchmark/build

amd-smi

LD_LIBRARY_PATH=~/.venv-mi210/lib/python3.10/site-packages/_rocm_sdk_devel/lib/ CSV_PREFIX="mi210" ../scripts/launch-bench.sh
