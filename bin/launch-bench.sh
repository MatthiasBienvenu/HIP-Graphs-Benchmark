# programName [outputFmt] [numTrials] [length] [width] [pattern] [stride] [maxLength]
# 	outputFmt - program output, default=3 (see below)
# 	numTrials (per length)
# 	starting length of the topology
# 	width - width of the graph topology
# 	pattern - Structure of graph, default=0 (see below)
# 	stride - how to grow the length between each set of trials
# 	maxLength - maximum lenght to try

# outputFmt can be:
# 	0: this help message
# 	1: csv data headers
# 	2: per trial csv data
# 	3: csv data & headers
# 	4: csv data is printed and trials are averaged for each length
# 	5: csv data is printed and trials are averaged for each length and headers are printed

# Pattern can be:
# 	0: No interconnect between branches
# 	1: Adds an extra root node before the initial fork

OUTPUT_FMT=3
NUM_TRIALS=100
LENGTH=1
WIDTH=1
PATTERN=0
STRIDE=1
MAX_LENGTH=10

bin/graphPerfScaling "$OUTPUT_FMT" "$NUM_TRIALS" "$LENGTH" "$WIDTH" "$PATTERN" "$STRIDE" "$MAX_LENGTH" >straight-line.csv

WIDTH=4
bin/graphPerfScaling "$OUTPUT_FMT" "$NUM_TRIALS" "$LENGTH" "$WIDTH" "$PATTERN" "$STRIDE" "$MAX_LENGTH" >${WIDTH}-parallel-lines.csv
