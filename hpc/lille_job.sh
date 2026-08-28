#!/bin/bash

#OAR -q default
#OAR -l host=1/gpu=1,walltime=1:00:00
#OAR -p chifflot AND gpu_model='Tesla V100-PCIE-32GB'
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err

module load apptainer

cd ~/HIP-Graphs-Benchmark/build

apptainer run --nv ~/hip-cuda-sandbox nvidia-smi

export CSV_PREFIX="v100"
apptainer run --nv ~/hip-cuda-sandbox ../scripts/launch-bench.sh
