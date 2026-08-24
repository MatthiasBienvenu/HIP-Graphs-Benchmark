#!/bin/bash

#OAR -q default
#OAR -l host=1/gpu=1,walltime=1:00:00
#OAR -p chifflot
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err

module load apptainer

cd ~/HIP-Graphs-Benchmark/build

apptainer run --nv ~/hip-cuda-sandbox nvidia-smi

apptainer run --nv ~/hip-cuda-sandbox ../scripts/launch-bench.sh
