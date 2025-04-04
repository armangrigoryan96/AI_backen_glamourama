# Installation

apt-get update && apt-get install -y \
    wget \
    bzip2 \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

apt-get update && apt-get install -y \
    python3-dev \
    python3 \
    build-essential \
    gcc \
    curl

apt-get update && apt-get install -y python3-opencv
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

apt-get update && apt-get install -y \
    vim \
    curl \
    bash \
    && apt-get clean


# Install conda

wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh && \
    bash Anaconda3-2022.05-Linux-x86_64.sh -b -p /root/anaconda3 && \
    /root/anaconda3/bin/conda init bash


PATH="/root/anaconda3/bin:$PATH"

conda update -n base -c defaults conda



# Create the Conda environment from the environment.yml
conda env create -f environment.yml

# Run the code
conda activate tryon; sh run.sh