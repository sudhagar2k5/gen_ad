#!/bin/bash
echo "Testing CARLA binary..."
/home/sudhagar/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -RenderOffScreen -nosound -carla-rpc-port=30000 2>&1 &
CPID=$!
sleep 10
if kill -0 $CPID 2>/dev/null; then
    echo "CARLA running"
    kill $CPID
else
    echo "CARLA exited"
fi
