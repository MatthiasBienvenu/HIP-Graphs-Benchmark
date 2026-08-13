#pragma once
#include <hip/hip_runtime.h>
#include <format>
#include <stdexcept>

#define HIP_CHECK(expr)                                      \
    do {                                                     \
        hipError_t err = (expr);                             \
        if (err != hipSuccess) {                             \
            throw std::runtime_error(std::format(            \
                "{}:{}: {} failed: {}",                      \
                __FILE__,                                    \
                __LINE__,                                    \
                #expr,                                       \
                hipGetErrorString(err)));                    \
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
