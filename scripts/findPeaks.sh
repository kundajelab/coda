#!/usr/bin/env bash

bds ${1}/chipseq.bds \
       -out_dir ${2} \
       -histone \
       -tag1 ${3} \
       -ctl_tag1 ${4} \
       -callpeak macs2 \
       -species ${5} \
       -nth 2


