FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    git \
    vim \
    && apt-get clean

RUN ln -s /usr/bin/python3 /usr/bin/py

WORKDIR /workspace
COPY requirements.txt .
COPY ./src ./src
COPY ./.git ./.git
COPY ./.gitignore ./.gitignore
RUN py -m venv .venv
RUN . .venv/bin/activate && pip install -r requirements.txt
