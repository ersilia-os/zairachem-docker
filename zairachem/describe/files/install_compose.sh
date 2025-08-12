#!/usr/bin/env bash
# install-docker-compose.sh
# Installs the latest Docker Compose v1 standalone binary on Ubuntu

set -euo pipefail

# 1. Determine latest release tag from GitHub
echo "Fetching latest Docker Compose v1 release..."
COMPOSE_VERSION=$(curl --silent "https://api.github.com/repos/docker/compose/releases/latest" \
  | grep '"tag_name":' \
  | sed -E 's/.*"([^"]+)".*/\1/')

echo "Latest version is ${COMPOSE_VERSION}"

# 2. Download the binary
BINARY_URL="https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)"
DEST="/usr/local/bin/docker-compose"

echo "Downloading from ${BINARY_URL} to ${DEST}..."
sudo curl -L "${BINARY_URL}" -o "${DEST}"

# 3. Make it executable
echo "Setting executable permissions on ${DEST}..."
sudo chmod +x "${DEST}"

# 4. Verify installation
echo "Verifying installation..."
docker-compose --version

echo "Docker Compose ${COMPOSE_VERSION} installed successfully!"
