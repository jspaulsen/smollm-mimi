#!/bin/bash

set -e

echo "Updating package list..."
apt update

echo "Installing pkg-config, cmake, and nano..."
apt install -y pkg-config cmake nano

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "Setup completed successfully!"
