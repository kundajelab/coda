#!/usr/bin/env bash
# Takes in a wig file 
# Outputs a bigWig file
# Then deletes the wig file

wigPath=$1
bigWigPath=$2
chromSizesPath=$3

# . /etc/profile.d/modules.sh
# module load ucsc_tools/3.0.9

wigToBigWig ${wigPath} ${chromSizesPath} ${bigWigPath}

rm ${wigPath}
