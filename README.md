# HIP Graphs Performance Benchmark

---

This repository contains source code and results of a benchmark designed for AMD's HIP accross AMD and NVIDIA targets. It is adapted from [NVIDIA/cuda-samples](https://github.com/nvidia/cuda-samples)'s cudaGraphsPerfScaling performance benchmark. I ported it to HIP and added some functionality to it to be able to compare performance of graphs accross multiples backends, brands and architectures. This work is a part of an research internship at SAMOVAR lab, Télécom SudParis.

---

## Project structure

The project is organized as follows:

- **`src/`**: Source code
- **`include/`**: Headers
- **`results/`**: All the data collected and details on the machines used
- **`scripts/`**: Scripts to launch the benchmark and display beautiful graphs
- **`apptainer/`**: Def files for a container running cuda 12.9 that can compile and run HIP programs

---

## Building from source

### Requirements

To build the project you need to be on a machine that has the _HIP runtime_ of your desired target (AMD or NVIDIA). It is relatively straight forward for AMD because it is almost always installed alongside ROCM. For nvidia you need to install the HIP NVIDIA runtime.

#### If you can be root on your system, then you should use your package manager.

For AMD follow [this guide](https://rocm.docs.amd.com/en/latest/install/rocm.html) to install the ROCM Core SDK.

For NVIDIA follow [this guide](https://rocm.docs.amd.com/projects/HIP/en/docs-5.7.1/how_to_guides/install.html).

If you don't find the packages you want you should probably check the [AMD ROCM repo](https://repo.radeon.com/rocm/).

#### If you cannot be root, for example on an HPC cluster, I still have some workarounds.

For AMD you can just install the ROCM SDK using pip in a venv since ROCM 7 so you can still follow [this guide](https://rocm.docs.amd.com/en/latest/install/rocm.html).

For NVIDIA, you cannot just use pip but if you have _apptainer_ installed, you can use [](apptainer/hip-cuda-12.9.def).

```
apptainer build apptainer/hip-cuda-12.9.sif apptainer/hip-cuda-12.9.def
apptainer run --nv apptainer/hip-cuda-12.9.sif
```

If you don't have the rights the build the container, you can build it on your local machine and then `scp` the `.sif` file to the remote cluster. If you don't have the right the run a `.sif` container on the cluster then you can still build the container with the `--sandbox` option which essentially builds it as a directory.

```
# On the local machine
apptainer build --sandbox apptainer/hip-cuda-12.9 apptainer/hip-cuda-12.9.def

# On the remote cluster
apptainer run --nv apptainer/hip-cuda-12.9
```

### Building the project

This project uses CMake so to build it you need to type the following commands

```
mkdir build
cd build
cmake ..
cmake --build .
```

---

## Running the benchmark

Stay in the `build` directory.

```
../scripts/launch-bench.sh
```

It will produce the results as `csv` files in the currently working directory.

This can take some time.

---

## Visualizing the results

To vizualize the results you can use [](scripts/graph.py).

```bash
python scripts/graph.py --help

# usage: graph.py [-h] [-t] [-c COLUMNS [COLUMNS ...]] [-x X_AXIS] filenames [filenames ...]

# Plot mean/std graphs from a CSV file, grouped by length.

# positional arguments:
#   filenames             Path(s) to the CSV file(s). Provide more than one to compare them on the same graphs.

# options:
#   -h, --help            show this help message and exit
#   -t, --transpose       Plot one graph per file instead of per column
#   -c, --columns COLUMNS [COLUMNS ...]
#                         Columns to plot (space separated). If omitted, all default columns are plotted.
#   -x, --x-axis X_AXIS   Column to use for the x axis.

python scripts/graph.py 4-parallel-lines.csv straight-line.csv
python scripts/graph.py 4-parallel-lines.csv straight-line.csv -c repeat_launch_total
python scripts/graph.py bin-tree -c repeat_launch_total baseline_launch_total -t
python scripts/graph.py *.csv -c repeat_launch_total -x nodes
```

---

## Metrics

blockingKernelTimeoutDetected 

### Topology parameters

The benchmark saves some static information about the current topology.
These static metrics are mainly used as the x axis for [](scripts/graph.py).

| Metrics Name |
| ------------ |
| length       |
| width        |
| nodes        |
| edges        |

### Baseline full stream execution (no graph)

| Metric Name            | Meaning                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------ |
| baseline_launch_api    | _CPU_ duration of sending all the asynchronous calls in the streams                  |
| baseline_launch_device | **GPU** duration of the execution                                                    |
| baseline_launch_total  | _CPU_ duration between the begining of the transmission and the end of the execution |

### Gaph construction and execution

| Metric Name          | Meaning                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| capture              | _CPU_ duration of capturing the streams and creating the graph                                               |
| instantiation        | _CPU_ duration of the graph instantiation                                                                    |
| first_launch_api     | _CPU_ duration of the first API call to hipGraphLaunch                                                       |
| first_launch_device  | **GPU** duration of the first execution of the graph                                                         |
| first_launch_total   | _CPU_ duration between the begining of the first call to hipGraphLaunch and the end of the first execution   |
| repeat_launch_api    | _CPU_ duration of the second API call to hipGraphLaunch                                                      |
| repeat_launch_total  | _CPU_ duration between the begining of the second call to hipGraphLaunch and the end of the second execution |
| repeat_launch_device | **GPU** duration of the second execution of the graph                                                        |
| upload_api           | _CPU_ duration of the API call to hipGraphUpload                                                             |
| upload_device        | **GPU** duration of the hipGraphUpload (mainly memory allocation and mapping)                                |

### Graph update

| Metric Name                            | Meaning                                                                                   |
| -------------------------------------- | ----------------------------------------------------------------------------------------- |
| single_node_update_api                 | _CPU_ duration of the API call to hipGraphExecKernelNodeSetParams                         |
| launch_after_single_node_update_device | **GPU** duration of the first graph launch after the single node update without uploading |
| full_graph_update_api                  | _CPU_ duration of the API call to hipGraphExecKernelNodeSetParams                         |
| launch_after_full_graph_update_device  | **GPU** duration of the first graph launch after the single node update without uploading |
