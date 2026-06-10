PROXY="http://10.93.144.53:8080"

# Add APT proxy (temporary for this session)
echo "Acquire::http::Proxy \"$PROXY\";" | sudo tee /etc/apt/apt.conf.d/95proxies
echo "Acquire::https::Proxy \"$PROXY\";" | sudo tee -a /etc/apt/apt.conf.d/95proxies

# Install NVIDIA container toolkit repo key
curl -fsSL -x $PROXY https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository list
curl -s -L -x $PROXY https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-container-toolkit runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker

# Test GPU container (with proxy env passed inside container)
docker run --rm --gpus all \
  -e HTTP_PROXY=$PROXY \
  -e HTTPS_PROXY=$PROXY \
  nvidia/cuda:12.0.0-base-ubuntu22.04 \
  nvidia-smi