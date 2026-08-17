/* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This file is a port of NVIDIA's original implementation to AMD HIP,
 * with additional modifications and extensions.
 *
 * This project is not affiliated with or endorsed by NVIDIA Corporation.
 */

//#define USE_ROCTX

#include <cassert>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>
#include <chrono>
#include "graphs.hip.hpp"

#ifdef __HIP_PLATFORM_AMD__
#include <hip/amd_detail/amd_hip_runtime.h>
// Used for AMDGPU to compute durations from number of ticks
__constant__ int d_wallClockRateKHz;
#endif

typedef volatile int LatchType;

std::chrono::time_point<std::chrono::high_resolution_clock> getCpuTime()
{
    return std::chrono::high_resolution_clock::now();
}

template <typename T>
float getMicroSecondDuration(T start, T end)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() *.001f;
}

float getAsyncMicroSecondDuration(hipEvent_t start, hipEvent_t end)
{
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, start, end));
    return ms*1000;
}

#ifdef USE_ROCTX
#include <rocprofiler-sdk-roctx/roctx.h>

class Tracer {
public:
    Tracer(const char* name) {
        roctxRangePushA(name);
    }
    ~Tracer() {
        roctxRangePop();
    }
};
#define RANGE(name) Tracer uniq_name_using_macros(name);
#define RANGE_PUSH(name) roctxRangePush(name)
#define RANGE_POP() roctxRangePop()
#else
#define RANGE(name)
#define RANGE_PUSH(name)
#define RANGE_POP()
#endif

std::vector<hipStream_t> stream;
std::vector<hipEvent_t> event;
hipEvent_t timingEvent[2];

struct hostData {
    long long timeElapsed;
    bool timeoutDetected;
    long long timeElapsed2;
    bool timeoutDetected2;
    LatchType latch;
    LatchType latch2;
};

struct hostData *hostData;

// Function to read the GPU nanosecond timer in a kernel
// wallClockRateHz is only used if on AMD GPU because we want nanoseconds but wall_clock64() gives raw ticks
__device__ __forceinline__ unsigned long long   __globaltimer() {
    #ifdef __HIP_PLATFORM_AMD__
        return (wall_clock64() * 1000000ULL) / (unsigned long long)d_wallClockRateKHz;
    #else
        unsigned long long globaltimer;
        asm volatile ("mov.u64 %0, %globaltimer;" : "=l"(globaltimer));
        return globaltimer;
    #endif
}

__global__ void waitWithTimeout(long long nanoseconds, bool* timeoutDetected, long long *timeElapsed, LatchType* latch)
{
    long long startTime = __globaltimer();
    long long endTime = startTime + nanoseconds;
    long long time = 0;
    do {
        time = __globaltimer();
    } while (time < endTime && (latch == NULL || *latch == 0));
    if (timeElapsed != NULL) {
        *timeElapsed = time - startTime;
    }
    if (timeoutDetected) {
        // report timeout if latch not detected
        *timeoutDetected = (latch == NULL || *latch == 0);
    }
}

__global__ void preUploadAnnotation()
{
}

__global__ void postUploadAnnotation()
{
}

std::vector<const char*> metricName;
std::vector<float> metricValue;

int counter2 = 0;
void runDemo(int length, int width, Topology topology)
{
    hipGraph_t graph;
    hipGraphExec_t graphExec;
    long long maxTimeoutNanoSeconds;
    {
        auto start = getCpuTime();
        graph = createGraph(length, width, topology);
        auto end = getCpuTime();
        metricName.push_back("capture");
        metricValue.push_back(getMicroSecondDuration(start, end));
    }
    {
        size_t nodes;
        size_t edges;

        HIP_CHECK(hipGraphGetNodes(graph, nullptr, &nodes));
        metricName.push_back("nodes");
        metricValue.push_back(nodes);

        HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &edges));
        metricName.push_back("edges");
        metricValue.push_back(edges);

        maxTimeoutNanoSeconds = 20000 + 5000 * (nodes + edges);
    }
    {
        RANGE("baseline stream execution");
        auto start = getCpuTime();

        // Launch topology directly (no capture)
        runGraphTopology(length, width, topology, 0);
        auto apiReturn = getCpuTime();

        HIP_CHECK(hipStreamSynchronize(stream[0]));
        auto streamSync = getCpuTime();

        metricName.push_back("baseline_launch_api");
        metricValue.push_back(getMicroSecondDuration(start, apiReturn));
        metricName.push_back("baseline_launch_total");
        metricValue.push_back(getMicroSecondDuration(start, streamSync));
    }
    {
        RANGE("baseline device execution in busy stream");
        waitWithTimeout<<<1,1,0,stream[0]>>>(maxTimeoutNanoSeconds, &hostData->timeoutDetected, &hostData->timeElapsed, &hostData->latch);

        HIP_CHECK(hipEventRecord(timingEvent[0], stream[0]));

        // launch topology directly into the blocked stream
        runGraphTopology(length, width, topology, 0);

        HIP_CHECK(hipEventRecord(timingEvent[1], stream[0]));

        // release the latch so execution begins
        hostData->latch = 1;
        HIP_CHECK(hipStreamSynchronize(stream[0]));

        metricName.push_back("baseline_launch_device");
        metricValue.push_back(getAsyncMicroSecondDuration(timingEvent[0], timingEvent[1]));

        hostData->latch = 0;
        hostData->timeoutDetected = 0;
    }
    {
        auto start = getCpuTime();
        HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));
        auto end = getCpuTime();

        metricName.push_back("instantiation");
        metricValue.push_back(getMicroSecondDuration(start, end));
    }
    {
        RANGE("launch including upload");

        auto start = getCpuTime();

        HIP_CHECK(hipGraphLaunch(graphExec, stream[0]));
        auto apiReturn = getCpuTime();

        HIP_CHECK(hipStreamSynchronize(stream[0]));
        auto streamSync = getCpuTime();

        metricName.push_back("first_launch_api");
        metricValue.push_back(getMicroSecondDuration(start, apiReturn));
        metricName.push_back("first_launch_total");
        metricValue.push_back(getMicroSecondDuration(start, streamSync));
    }
    {
        RANGE("repeat lauch in empty stream");

        auto start = getCpuTime();

        HIP_CHECK(hipGraphLaunch(graphExec, stream[0]));
        auto apiReturn = getCpuTime();

        HIP_CHECK(hipStreamSynchronize(stream[0]));
        auto streamSync = getCpuTime();

        metricName.push_back("repeat_launch_api");
        metricValue.push_back(getMicroSecondDuration(start, apiReturn));
        metricName.push_back("repeat_launch_total");
        metricValue.push_back(getMicroSecondDuration(start, streamSync));
    }
    {
        // re-instantiating the exec to simulate first launch into a busy stream.
        HIP_CHECK(hipGraphExecDestroy(graphExec));
        HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));

        waitWithTimeout<<<1,1,0,stream[0]>>>(maxTimeoutNanoSeconds, &hostData->timeoutDetected, &hostData->timeElapsed, &hostData->latch);

        RANGE("launch including upload in busy stream");
        HIP_CHECK(hipEventRecord(timingEvent[0], stream[0]));
        HIP_CHECK(hipGraphLaunch(graphExec, stream[0]));
        HIP_CHECK(hipEventRecord(timingEvent[1], stream[0]));

        hostData->latch = 1;
        HIP_CHECK(hipStreamSynchronize(stream[0]));

        metricName.push_back("first_launch_device");
        metricValue.push_back(getAsyncMicroSecondDuration(timingEvent[0], timingEvent[1]));
        metricName.push_back("blockingKernelTimeoutDetected");
        metricValue.push_back(hostData->timeoutDetected);
        hostData->latch = 0;
        hostData->timeoutDetected = 0;
    }
    {
        RANGE("repeat lauch in busy stream");
        waitWithTimeout<<<1,1,0,stream[0]>>>(maxTimeoutNanoSeconds, &hostData->timeoutDetected, &hostData->timeElapsed, &hostData->latch);
        HIP_CHECK(hipEventRecord(timingEvent[0], stream[0]));
        HIP_CHECK(hipGraphLaunch(graphExec, stream[0]));
        HIP_CHECK(hipEventRecord(timingEvent[1], stream[0]));

        hostData->latch = 1;
        HIP_CHECK(hipStreamSynchronize(stream[0]));

        metricName.push_back("repeat_launch_device");
        metricValue.push_back(getAsyncMicroSecondDuration(timingEvent[0], timingEvent[1]));
        metricName.push_back("blockingKernelTimeoutDetected");
        metricValue.push_back(hostData->timeoutDetected);
        hostData->latch = 0;
        hostData->timeoutDetected = 0;
    }
    {
        // re-instantiating the exec to provide upload with work to do.
        HIP_CHECK(hipGraphExecDestroy(graphExec));
        HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));
        waitWithTimeout<<<1,1,0,stream[0]>>>(maxTimeoutNanoSeconds, &hostData->timeoutDetected2, &hostData->timeElapsed2, &hostData->latch2);
        waitWithTimeout<<<1,1,0,stream[1]>>>(maxTimeoutNanoSeconds, &hostData->timeoutDetected, &hostData->timeElapsed, &hostData->latch);

        RANGE("uploading a graph off of the critical path");
        preUploadAnnotation<<<1,1,0,stream[1]>>>();
        HIP_CHECK(hipEventRecord(timingEvent[0], stream[0]));
        auto start = getCpuTime();

        HIP_CHECK(hipGraphUpload(graphExec, stream[1]));

        auto apiReturn = getCpuTime();
        HIP_CHECK(hipEventRecord(event[0],stream[1]));
        HIP_CHECK(hipEventRecord(timingEvent[1], stream[0]));
        postUploadAnnotation<<<1,1,0,stream[1]>>>();

        hostData->latch = 1; // release the blocking kernel for the upload
        HIP_CHECK(hipStreamWaitEvent(stream[0], event[0], 0));
        HIP_CHECK(hipGraphLaunch(graphExec, stream[0]));
        HIP_CHECK(hipEventSynchronize(event[0])); // upload done, similuate critical path being ready for the graph to run by the release of the second latch

        hostData->latch2 = 1; // release the work
        HIP_CHECK(hipStreamSynchronize(stream[0]));

        metricName.push_back("upload_api");
        metricValue.push_back(getMicroSecondDuration(start, apiReturn));
        metricName.push_back("upload_device");
        metricValue.push_back(getAsyncMicroSecondDuration(timingEvent[0], timingEvent[1]));
        metricName.push_back("blockingKernelTimeoutDetected");
        metricValue.push_back(hostData->timeoutDetected);

        hostData->latch = 0;
        hostData->latch2 = 0;
        hostData->timeoutDetected = 0;
        hostData->timeoutDetected2 = 0;
    }
    {
        RANGE("update a single node on busy stream");

        hipGraphNode_t node;
        size_t numNodes = 1;
        hipKernelNodeParams params;
        hipGraphNodeType type;

        HIP_CHECK(hipGraphGetNodes(graph, &node, &numNodes));

        // make sure it's a kernel node
        HIP_CHECK(hipGraphNodeGetType(node, &type));

        assert(type == hipGraphNodeTypeKernel);

        HIP_CHECK(hipGraphKernelNodeGetParams(node, &params));

        // change the argument of a single kernel node to a random value
        int unused = 42;
        void *args[] = {
            &unused
        };

        params.kernelParams = args;

        auto apiStart = getCpuTime();
        HIP_CHECK(hipGraphExecKernelNodeSetParams(graphExec, node, &params));
        auto apiEnd = getCpuTime();

        waitWithTimeout<<<1,1,0,stream[0]>>>(maxTimeoutNanoSeconds, &hostData->timeoutDetected, &hostData->timeElapsed, &hostData->latch);

        HIP_CHECK(hipEventRecord(timingEvent[0], stream[0]));
        HIP_CHECK(hipGraphLaunch(graphExec, stream[0]));
        HIP_CHECK(hipEventRecord(timingEvent[1], stream[0]));

        hostData->latch = 1;
        HIP_CHECK(hipStreamSynchronize(stream[0]));

        metricName.push_back("single_node_update_api");
        metricValue.push_back(getMicroSecondDuration(apiStart, apiEnd));
        metricName.push_back("launch_after_single_node_update_device");
        metricValue.push_back(getAsyncMicroSecondDuration(timingEvent[0], timingEvent[1]));
        metricName.push_back("blockingKernelTimeoutDetected");
        metricValue.push_back(hostData->timeoutDetected);


        hostData->latch = 0;
        hostData->timeoutDetected = 0;

        // reset the graph and launch it
        unused = 0;
        HIP_CHECK(hipGraphExecKernelNodeSetParams(graphExec, node, &params));
        HIP_CHECK(hipGraphLaunch(graphExec, stream[0]));
        HIP_CHECK(hipStreamSynchronize(stream[0]));
    }
    {
        RANGE("update the full graph on busy stream");

        // create a graph with a random argument for the kernel
        hipGraph_t graph2 = createGraph(length, width, topology, -67);
        hipGraphNode_t errNode;
        hipGraphExecUpdateResult result;

        auto apiStart = getCpuTime();
        HIP_CHECK(hipGraphExecUpdate(graphExec, graph2, &errNode, &result));
        auto apiEnd = getCpuTime();

        waitWithTimeout<<<1,1,0,stream[0]>>>(maxTimeoutNanoSeconds, &hostData->timeoutDetected, &hostData->timeElapsed, &hostData->latch);

        HIP_CHECK(hipEventRecord(timingEvent[0], stream[0]));
        HIP_CHECK(hipGraphLaunch(graphExec, stream[0]));
        HIP_CHECK(hipEventRecord(timingEvent[1], stream[0]));

        hostData->latch = 1;
        HIP_CHECK(hipStreamSynchronize(stream[0]));

        metricName.push_back("full_graph_update_api");
        metricValue.push_back(getMicroSecondDuration(apiStart, apiEnd));
        metricName.push_back("launch_after_full_graph_update_device");
        metricValue.push_back(getAsyncMicroSecondDuration(timingEvent[0], timingEvent[1]));
        metricName.push_back("blockingKernelTimeoutDetected");
        metricValue.push_back(hostData->timeoutDetected);

        hostData->latch = 0;
        hostData->timeoutDetected = 0;

        // reset the graph and launch it
        HIP_CHECK(hipGraphDestroy(graph2));
        HIP_CHECK(hipGraphExecUpdate(graphExec, graph, &errNode, &result));
        HIP_CHECK(hipGraphLaunch(graphExec, stream[0]));
        HIP_CHECK(hipStreamSynchronize(stream[0]));
    }
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
    RANGE_POP();
}

void usage() {
    printf("programName [outputFmt] [numTrials] [length] [width] [topology] [stride] [maxLength] \n");
    printf("\toutputFmt - program output, default=3 (see below)\n");
    printf("\tnumTrials (per length)\n");
    printf("\tstarting length of the topology\n");
    printf("\twidth - width of the graph topology\n");
    printf("\ttopology - Structure of graph, default=0 (see below)\n");
    printf("\tstride - how to grow the length between each set of trials \n");
    printf("\tmaxLength - maximum lenght to try \n");
    printf("\n");
    printf("outputFmt can be:\n");
    printf("\t0: this help message\n");
    printf("\t1: csv data headers\n");
    printf("\t2: per trial csv data\n");
    printf("\t3: csv data & headers\n");
    printf("\t4: csv data is printed and trials are averaged for each length\n");
    printf("\t5: csv data is printed and trials are averaged for each length and headers are printed\n");
    printf("\n");
    printf("Topology can be:\n");
    printf("\t0: ParallelChains. <width> independent parallel lines of size <length> \n");
    printf("\t1: ParallelChainsSingleEntry. Same as 0 but with a additional root node\n");
    printf("\t2: MeshGraph. <length> layers of size <width> where each node is connected to 2 nodes in the next layer\n");
    printf("\t3: MeshGraphSingleEntry. Same as 2 but with a additional root node\n");
    printf("\t4: BinaryTree of depth <length> (the number of nodes is n*(n+1)/2). Width will be ignored.\n");
}

int main(int argc, char **argv)
{
    if(argc < 1) {
        usage();
        return 0;
    }

    unsigned int numTrials=1, length=20, width=1, outputFmt=3, topo=0, stride = 1;
    if(argc > 1) outputFmt = atoi(argv[1]);
    if(argc > 2) numTrials = atoi(argv[2]);
    if(argc > 3) length= atoi(argv[3]);
    if(argc > 4) width= atoi(argv[4]);
    if(argc > 5) topo = atoi(argv[5]);
    if(argc > 6) stride = atoi(argv[6]);
    int maxLength = length;
    if(argc > 7) maxLength = atoi(argv[7]);
    if (maxLength < length) {
        maxLength = length;
    }

    if((outputFmt & 4) && (outputFmt & 2)) {
        printf("printing average and all samples doesn't make sense\n");
    }

    if(length == 0 ||
       width == 0 ||
       outputFmt == 0 ||
       outputFmt > 5 ||
       topo > 4)
    {
        usage();
        return 0;
    }

    auto topology{static_cast<Topology>(topo)};

    if (topology == Topology::BinaryTree) {
        width = 1 << (maxLength - 1);
    }

    hipGraph_t graph;

    HIP_CHECK(hipFree(0));

    #ifdef __HIP_PLATFORM_AMD__
    int wallClockRateKHz = 0;
    HIP_CHECK(hipDeviceGetAttribute(&wallClockRateKHz, hipDeviceAttributeWallClockRate, 0));
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_wallClockRateKHz), &wallClockRateKHz, sizeof(wallClockRateKHz)));
    #endif

    HIP_CHECK(hipHostMalloc(&hostData, sizeof(*hostData)));

    int numStreams = width;
    if (numStreams == 1)
        numStreams = 2; // demo needs two streams even if capture only needs 1.

    stream.resize(numStreams);
    event.resize(width); // may not use the whole size depending on topology

    for (int i = 0; i < numStreams; i++)
    {
        HIP_CHECK(hipStreamCreate(&stream[i]));
    }

    for (int i = 0; i < width; i++) {
        HIP_CHECK(hipEventCreateWithFlags(&event[i], hipEventDisableTiming));
    }

    HIP_CHECK(hipEventCreate(&timingEvent[0]));
    HIP_CHECK(hipEventCreate(&timingEvent[1]));

    {
        RANGE("warmup");
        for (int i = 0; i < numStreams; i++)
        {
            empty<<<1,1,0,stream[i]>>>(0);
        }
        HIP_CHECK(hipStreamSynchronize(stream[0]));

        runDemo(length, width, topology);
    }

    if (outputFmt & 1) {
        printf("length, width, pattern, ");
        for (int i = 0; i < metricName.size(); i++) {
            printf("%s, ", metricName[i]);
        }
        printf("\r\n");
    }

    if (!(outputFmt & 6)) {
        printf("skipping trials since no output is expected\n");
        return 0;
    }

    std::vector<double> metricTotal;
    metricTotal.resize(metricValue.size());

    while (length <= maxLength) {
        for (int i = 0; i < numTrials; i++) {
            metricName.clear();
            metricValue.clear();

            runDemo(length, width, topology);

            if (outputFmt & 2) {
                printf("%d, %d, %d, ",length, width, topo);
                for (int i = 0; i < metricValue.size(); i++) {
                    printf("%0.3f, ", metricValue[i]);
                }
                printf("\r\n");
            }
            if (outputFmt & 4) {
                for (int i = 0; i < metricTotal.size(); i++) {
                    metricTotal[i] += metricValue[i];
                }
            }
        }

        if (outputFmt & 4) {
            printf("%d, %d, %d, ",length, width, topo);
            for (int i = 0; i < metricTotal.size(); i++) {
                printf("%0.3f, ", metricTotal[i]/numTrials);
                metricTotal[i] = 0;
            }
            printf("\r\n");
        }

        length += stride;
    }

    printf("\n");
    return 0;
}
