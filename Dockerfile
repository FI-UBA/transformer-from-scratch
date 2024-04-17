# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04
# FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Set non-interactive option
ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video

# Install dependencies
RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y git wget curl vim build-essential cmake unzip pkg-config mc sudo

RUN useradd -m -u 1000 -G sudo -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Change user
USER vscode

# This is to avoid Python packages installation warning
RUN mkdir -p /home/vscode/.local/bin

RUN sudo apt-get update && sudo apt-get -y install python3.10 python3.10-venv python3-pip && \
    sudo apt-get -y clean

# Expose port 6006 for Tensorboard
EXPOSE 6006

# Bash
CMD [ "/bin/bash" ]
