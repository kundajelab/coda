#!/usr/bin/env bash
# Takes in a bigWig file and BED file
# Gets the average of the bigWig signal over the BED intervals
# Outputs it to outputPath

bigWigPath=$1
BEDPath=$2
outputPath=$3

# . /etc/profile.d/modules.sh
# module load ucsc_tools/3.0.9

# bigWigAverageOverBed has no option to output to stdout, so we need a temp file
bigWigAverageOverBed ${bigWigPath} ${BEDPath} ${outputPath}.temp

cut -f5 ${outputPath}.temp > ${outputPath}
rm ${outputPath}.temp


