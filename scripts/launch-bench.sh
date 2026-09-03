# programName [outputFmt] [numTrials] [length] [width] [topology] [stride] [maxLength]
# 	outputFmt - program output, default=3 (see below)
# 	numTrials (per length)
# 	starting length of the topology
# 	width - width of the graph topology
# 	topology - Structure of graph, default=0 (see below)
# 	stride - how to grow the length between each set of trials
# 	maxLength - maximum lenght to try

# outputFmt can be:
# 	0: this help message
# 	1: csv data headers
# 	2: per trial csv data
# 	3: csv data & headers
# 	4: csv data is printed and trials are averaged for each length
# 	5: csv data is printed and trials are averaged for each length and headers are printed

# Topology can be:
# 	0: ParallelChains. <width> independent parallel lines of size <length>
# 	1: ParallelChainsSingleEntry. Same as 0 but with a additional root node
# 	2: MeshGraph. <length> layers of size <width> where each node is connected to 2 nodes in the next layer
# 	3: MeshGraphSingleEntry. Same as 2 but with a additional root node
# 	4: BinaryTree of depth <length> (the number of nodes is n*(n+1)/2). Width will be ignored.

OUTPUT_FMT=3
NUM_TRIALS=100
STRIDE=10

echo "Testing single line graph"

LENGTH=10
WIDTH=1
PATTERN=0
MAX_LENGTH=2000
./graphPerfScaling "$OUTPUT_FMT" "$NUM_TRIALS" "$LENGTH" "$WIDTH" "$PATTERN" "$STRIDE" "$MAX_LENGTH" >"${CSV_PREFIX}_straight-line.csv"

echo "Testing parallel lines graph"

LENGTH=10
WIDTH=4
PATTERN=0
MAX_LENGTH=2000
./graphPerfScaling "$OUTPUT_FMT" "$NUM_TRIALS" "$LENGTH" "$WIDTH" "$PATTERN" "$STRIDE" "$MAX_LENGTH" >"${CSV_PREFIX}_${WIDTH}-parallel-lines.csv"

echo "Testing mesh graph"

LENGTH=10
WIDTH=10
PATTERN=2
MAX_LENGTH=100
./graphPerfScaling "$OUTPUT_FMT" "$NUM_TRIALS" "$LENGTH" "$WIDTH" "$PATTERN" "$STRIDE" "$MAX_LENGTH" >"${CSV_PREFIX}_${WIDTH}-mesh-graph.csv"

echo "Testing bintree graph"

LENGTH=1
STRIDE=1
PATTERN=4
MAX_LENGTH=10
./graphPerfScaling "$OUTPUT_FMT" "$NUM_TRIALS" "$LENGTH" "$WIDTH" "$PATTERN" "$STRIDE" "$MAX_LENGTH" >"${CSV_PREFIX}_bin-tree.csv"
