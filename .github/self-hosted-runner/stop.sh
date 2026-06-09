#!/bin/bash
set -euo pipefail

# Stop all running stablebear-runner containers and clean up.
# Usage: ./stop.sh

# Kill run.sh first so it doesn't respawn containers
if pkill -f "run\.sh"; then
    echo "Killed run.sh background processes."
fi

echo "Stopping stablebear-runner containers..."
CONTAINERS=$(docker ps -q --filter "ancestor=stablebear-runner:cuda12")

if [[ -z "$CONTAINERS" ]]; then
    echo "No running stablebear-runner containers found."
else
    docker stop $CONTAINERS
    echo "Stopped."
fi
