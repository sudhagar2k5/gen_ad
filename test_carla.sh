#!/bin/bash
source /home/sudhagar/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo

echo "Starting CARLA..."
DISPLAY= /home/sudhagar/carla/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=30000 &
CPID=$!
echo "CARLA PID: $CPID"
sleep 20

if kill -0 $CPID 2>/dev/null; then
    echo "CARLA is running!"
    python -c "
import carla
try:
    client = carla.Client('localhost', 30000)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f'CARLA connected! Map: {world.get_map().name}')
except Exception as e:
    print(f'Connection failed: {e}')
"
    kill $CPID 2>/dev/null
else
    echo "CARLA failed to start"
fi
