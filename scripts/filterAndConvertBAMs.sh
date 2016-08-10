#!/usr/bin/env bash
# Takes in a BAM file 
# Filters it for properly mapping reads above a certain MAPQ threshold 
# Returns BEDPE

BAMPath=$1
BEDPath=$2
mapQThreshold=$3

# . /etc/profile.d/modules.sh
# module load samtools/1.2
# module load bedtools/2.23.0

samtools view -F 1804 -f 2 -q ${mapQThreshold} -u ${BAMPath} | \
    samtools sort -m 10000M -O bam -n -T ${BAMPath} - | \
    samtools fixmate -r -O 'bam' - - | \
    samtools view -F 1804 -f 2 -u - | \
    bedtools bamtobed -bedpe -i stdin > ${BEDPath}