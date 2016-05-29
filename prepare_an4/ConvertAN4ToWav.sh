#!/bin/bash
#Place inside the an4 directory to convert the raw samples into wav.
for entry in "$1/wav"
do
	for wav_folder in "$entry"/*
	do
		for sampleFolder in "$wav_folder"/*
			do
				for sample in "$sampleFolder"/*
					do
					sox -t raw -r 16000 -b 16 -e signed-integer -B -c 1  "$sample" "${sample%.*}.wav"
					done
			done
	done
done
