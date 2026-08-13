#include <hip/hip_runtime.h>
#include "graphs.hip.hpp"

extern std::vector<hipEvent_t> event;
extern std::vector<hipStream_t> stream;

__global__ void empty(int unused)
{
}

void runParallelChain(int length, int width, Topology topology, int kernelArg)
{
    if (topology == Topology::ParallelChainsSingleEntry) {
       empty<<<1,1,0,stream[0]>>>(kernelArg);
    }

    HIP_CHECK(hipEventRecord(event[0], stream[0]));
    for (int i = 1; i < width; i++) {
        HIP_CHECK(hipStreamWaitEvent(stream[i], event[0], 0));
    }

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < length; j++) {
            empty<<<1,1,0,stream[i]>>>(kernelArg);
        }
    }

    for (int i = 1; i < width; i++) {
        HIP_CHECK(hipEventRecord(event[0], stream[i]));
        HIP_CHECK(hipStreamWaitEvent(stream[0], event[0], 0));
    }
}

void runMeshGraph(unsigned int length, unsigned int width, Topology topology, int kernelArg)
{
    if (topology == Topology::MeshGraphSingleEntry) {
        empty<<<1, 1, 0, stream[0]>>>(kernelArg);
    }

    HIP_CHECK(hipEventRecord(event[0], stream[0]));

    for (int i = 1; i < width; i++) {
        HIP_CHECK(hipStreamWaitEvent(stream[i], event[0], 0));
    }

    for (unsigned int l {0}; l < length - 1; l++) {
        for (unsigned int w {0}; w < width; w++) {
            empty<<<1, 1, 0, stream[w]>>>(kernelArg);
            HIP_CHECK(hipEventRecord(event[w], stream[w]));
        }

        for (unsigned int w {0}; w < width; w++) {
            HIP_CHECK(hipStreamWaitEvent(stream[(w + 1) % width], event[w], 0));
        }
    }

    // last layer has special syncronization
    if (length >= 1) {
        for (unsigned int w {0}; w < width; w++) {
            empty<<<1, 1, 0, stream[w]>>>(kernelArg);
        }

        for (unsigned int w {1}; w < width; w ++) {
            HIP_CHECK(hipEventRecord(event[0], stream[w]));
            HIP_CHECK(hipStreamWaitEvent(stream[0], event[0]));
        }
    }
}

void runBinaryTree(unsigned int length, int kernelArg)
{
    size_t offset = stream.size();

    for (unsigned int depth {0}; depth < length - 1; depth++) {
        size_t new_offset = offset / 2;
        for (unsigned int i {0}; i < stream.size(); i += offset) {
            // the children of node i are i and i + new_offset
            empty<<<1, 1, 0, stream[i]>>>(kernelArg);
            HIP_CHECK(hipEventRecord(event[0], stream[i]));
            HIP_CHECK(hipStreamWaitEvent(stream[i + new_offset], event[0]));
        }
        offset = new_offset;
    }

    for (unsigned int i {0}; i < stream.size(); i++) {
        empty<<<1, 1, 0, stream[i]>>>(kernelArg);
        HIP_CHECK(hipEventRecord(event[0], stream[i]));
        HIP_CHECK(hipStreamWaitEvent(stream[0], event[0]));
    }
}

void runGraphTopology(unsigned int length, unsigned int width, Topology topology, int kernelArg)
{
    switch (topology) {
        case Topology::ParallelChains:
        case Topology::ParallelChainsSingleEntry:
            runParallelChain(length, width, topology, kernelArg);
            break;
        case Topology::MeshGraph:
        case Topology::MeshGraphSingleEntry:
            runMeshGraph(length, width, topology, kernelArg);
            break;
        case Topology::BinaryTree:
            runBinaryTree(length, kernelArg);
            break;
        default:
            throw std::logic_error{"Unreachable"};
    }
}

hipGraph_t createGraph(unsigned int length, unsigned int width, Topology topology, int kernelArg) {
    hipGraph_t graph;

    HIP_CHECK(hipStreamBeginCapture(stream[0], hipStreamCaptureModeGlobal));
    runGraphTopology(length, width, topology, kernelArg);
    HIP_CHECK(hipStreamEndCapture(stream[0], &graph));

    return graph;
}
