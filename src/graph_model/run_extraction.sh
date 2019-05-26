#!/bin/bash
i=0
for file in /subgraph/train/*
do
  BASENAME=$(basename $file)
  echo $BASENAME
  python extract_path.py /Users/debjit/Downloads/python_script/script/python_context2context.py --graph_path /Users/debjit/Downloads/data/storycommonsense_data/subgraph/train/$BASENAME* --purpose train
done
