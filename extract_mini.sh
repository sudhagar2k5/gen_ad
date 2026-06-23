#!/bin/bash
cd /home/sudhagar/Bench2DriveZoo/data/bench2drive/v1
for f in *.tar.gz; do
  echo "Extracting $f"
  tar xzf "$f" && rm "$f"
done
echo "ALL_EXTRACTED"
ls -d */ 2>/dev/null || echo "No directories"
