#!/bin/bash
i=0
for file in /subgraph/train/*
do
  BASENAME=$(basename $file)
  echo $BASENAME
  python extract_path.py --graph_path /Users/debjit/Downloads/data/storycommonsense_data/subgraph/train/$BASENAME* -input_path input.txt -output_path output.txt --purpose train
done
