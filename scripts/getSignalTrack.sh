#!/usr/bin/env bash
# Runs the BDS CHIP-seq pipeline on a given tagAlign file.

pipelineDir=$1
tagAlignPath=$2
outputDir=$3
species=$4

bds ${pipelineDir}/chipseq.bds \
    -out_dir ${outputDir} \
    -histone \
    -input tag \
    -final_stage xcor \
    -tag1 ${tagAlignPath} \
    -tag2bw \
    -species ${species} \
    -nth 2
