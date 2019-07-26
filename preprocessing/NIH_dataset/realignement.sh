#!/bin/bash
for file in pancreas_test/data/*
do
	echo "$file"
	n=${file##*/}
	fold=${file%/*}
	echo "$n"
	nam=${n##*_}
	echo "$nam"
	out="${fold}/${nam}"
	gzout="${fold}/${nam}.gz"


	fslswapdim "$file" x -y z "$out"
	gunzip "$gzout"
	rm "$gzout"
	rm "$file"
	fslorient -forceneurological "$out" 


done
