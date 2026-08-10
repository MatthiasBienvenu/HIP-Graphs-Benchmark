.PHONY: clean all run

all: bin/graphPerfScaling

run: bin/graphPerfScaling
	bin/launch-bench.sh

bin/graphPerfScaling: src/graphPerfScaling.hip
	hipcc $^ -o $@ -x cu -std=c++23
