#pragma once
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

#define HIP_CHECK(expr)                                      \
    do {                                                     \
        hipError_t err = (expr);                             \
        if (err != hipSuccess) {                             \
            std::cerr                                        \
                << __FILE__ << ":"                           \
                << __LINE__ << ": "                          \
                << (#expr) << " failed: "                    \
                << hipGetErrorString(err);                   \
            abort();                                        \
        }                                                    \
    } while (false)

enum class Topology {
    ParallelChains = 0,
    ParallelChainsSingleEntry,
    MeshGraph,
    MeshGraphSingleEntry,
    BinaryTree,
};

__global__ void empty(int unused);
hipGraph_t createGraph(unsigned int length, unsigned int width, Topology topology, int kernelArg = 0);
void runGraphTopology(unsigned int length, unsigned int width, Topology topology, int kernelArg);
