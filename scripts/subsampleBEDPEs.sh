#!/usr/bin/env bash
# Takes in a BEDPE file
# Subsamples it and outputs tagAlign
# numSamplePairs is measured in pairs of reads

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

BEDPath=$1
tagAlignPath=$2
numSamplePairs=$3

awkProg='
BEGIN {OFS = "\t"}
{
	printf "%s\t%s\t%s\tN\t1000\t%s\n%s\t%s\t%s\tN\t1000\t%s\n",$1,$2,$3,$9,$4,$5,$6,$10
}
'

shuf -n ${numSamplePairs} --random-source=<(get_seeded_random 42) ${BEDPath} | \
	awk -F'\t' "${awkProg}" | \
	gzip -c > ${tagAlignPath}