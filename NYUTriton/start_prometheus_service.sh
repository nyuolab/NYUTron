#!/bin/bash
# To access this, make sure to forward the hosting port 9090 in your SSH connection
# ssh USERNAME@IP -L 9090:127.0.0.1:9090 -L 3000:127.0.0.1:3000
# ./start_prometheus_service.sh
# This will also start Grafana running at 127.0.0.1:3000 to visualize results (default user/pass = admin/admin)
# On the data import dashboard you'll need to import http://127.0.0.1:9090
# Load board: 12832

PROD=$1

function docker_run() {
docker run \
    --name prometheus \
    -d \
    --ipc=host --net=host \
    -p ${PROMETHEUSPORT}:${PROMETHEUSPORT} \
    -v /home/oermae01/NYUtriton/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus

docker run \
    -d \
    --name=grafana \
    --ipc=host --net=host \
    -p ${GRAFANAPORT}:${GRAFANAPORT} \
    grafana/grafana-enterprise:8.5.4-ubuntu
}

# Main logic
if [ "${PROD}"=="--prod" ]; then
  PROMETHEUSPORT=9090
  GRAFANAPORT=3000
  docker_run
  echo "STARTED PROD MONITORING SERVICES"
else 
  PROMETHEUSPORT=9095
  GRAFANAPORT=3005
  docker_run
  echo "STARTED DEV MONITORING SERVICES"
fi