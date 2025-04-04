# Use the official NVIDIA CUDA 12.0 base image with Ubuntu
#FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:12.0.1-devel-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
# Install dependencies and Miniconda
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3 \
    build-essential \
    gcc \
    curl

RUN apt-get update && apt-get install -y python3-opencv
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

RUN apt-get update && apt-get install -y \
    vim \
    curl \
    bash \
    && apt-get clean


# Install Miniconda
#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
#    bash miniconda.sh -b -p /opt/conda && \
#    rm miniconda.sh

RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh && \
    bash Anaconda3-2022.05-Linux-x86_64.sh -b -p /root/anaconda3 && \
    /root/anaconda3/bin/conda init bash

# Add Conda to the PATH permanently
ENV PATH="/root/anaconda3/bin:$PATH"
RUN conda update -n base -c defaults conda


# Copy the environment.yml file into the container
COPY environment.yml /app/

# Create the Conda environment from the environment.yml
RUN conda env create -f environment.yml

# Activate the environment by default
SHELL ["/bin/bash", "-c"]
RUN echo "conda activate tryon" >> ~/.bashrc

# Copy the Flask app code into the container
COPY . /app

# Expose port for the Flask app to run
EXPOSE 5000

# Run the Flask app
#CMD ["conda", "run", "-n", "tryon", "python3", "tryon_dress/test_start.py"]
CMD ["sh", "run.sh"]

