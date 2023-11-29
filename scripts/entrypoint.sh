#!/bin/bash
set -e
source /startup.sh
sleep 2  # Make sure any services are up and running
exec "$@"
