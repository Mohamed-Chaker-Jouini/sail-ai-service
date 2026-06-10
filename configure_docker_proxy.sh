#!/bin/bash
# configure_proxy_docker.sh
# Configures Docker daemon proxy settings and restarts the service

set -e

# ---- Configuration ----
HTTP_PROXY="http://10.93.144.53:8080"
HTTPS_PROXY="http://10.93.144.53:8080"
NO_PROXY="localhost,127.0.0.1,vllm,10.0.0.0/8"

PROXY_DIR="/etc/systemd/system/docker.service.d"
PROXY_FILE="$PROXY_DIR/http-proxy.conf"

# ---- Check root ----
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or with sudo"
  exit 1
fi

echo "Creating directory: $PROXY_DIR"
mkdir -p "$PROXY_DIR"

echo "Writing proxy config to: $PROXY_FILE"
cat > "$PROXY_FILE" <<EOF
[Service]
Environment="HTTP_PROXY=${HTTP_PROXY}"
Environment="HTTPS_PROXY=${HTTPS_PROXY}"
Environment="NO_PROXY=${NO_PROXY}"
EOF

echo "Proxy config written:"
echo "-----------------------------"
cat "$PROXY_FILE"
echo "-----------------------------"

echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Restarting Docker service..."
systemctl restart docker

echo "Verifying proxy environment on Docker service..."
systemctl show --property=Environment docker

echo "Done. You can now retry: sudo docker compose up --build"