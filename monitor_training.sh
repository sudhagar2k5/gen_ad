#!/bin/bash
# Run this to monitor GenAD training progress
# Usage: wsl -d Ubuntu-22.04 then: bash /mnt/c/GenAD/monitor_training.sh

source /home/sudhagar/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo

LOG_JSON=$(ls -t /home/sudhagar/Bench2DriveZoo/work_dirs/GenAD_config_b2d/*.log.json 2>/dev/null | head -1)

echo "=== GPU Status ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader

echo ""
echo "=== Latest Training Metrics ==="
if [ -f "$LOG_JSON" ]; then
    tail -3 "$LOG_JSON" | python -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        if d.get('mode') == 'train':
            print(f\"Epoch {d['epoch']}, Iter {d['iter']}: loss={d['loss']:.4f}, lr={d['lr']:.6f}, time={d['time']:.2f}s\")
    except:
        pass
"
else
    echo "No log file found yet"
fi

echo ""
echo "=== Checkpoints ==="
ls -lh /home/sudhagar/Bench2DriveZoo/work_dirs/GenAD_config_b2d/*.pth 2>/dev/null || echo "No checkpoints yet"
