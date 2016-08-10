#!/usr/bin/env bash
# Takes in a BAM file with single-end reads
# Filters it for properly mapping reads above a certain MAPQ threshold 
# Returns BED

BAMPath=$1
BEDPath=$2
mapQThreshold=$3

# . /etc/profile.d/modules.sh
# module load samtools/1.2
# module load bedtools/2.23.0

samtools view -F 1804 -q ${mapQThreshold} -u ${BAMPath} | \
    samtools sort -m 10000M -O bam -n -T ${BAMPath} - | \
    samtools view -F 1804 -u - | \
    bedtools bamtobed -i stdin > ${BEDPath}