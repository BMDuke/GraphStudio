# Build from base tensorflow image
FROM nvidia/cuda:11.0-base

# CMD nvidia-smi

# Clean up nvidia installation
RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Install python 
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

# Set up environment
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir graph-studio
WORKDIR /graph-studio

COPY app/cache cache
COPY app/conf conf
COPY app/data data
COPY app/source source
COPY app/tests tests

# Define aliases for CLI
RUN echo 'alias fetch="python3 -m source.cli.download"' >> ~/.bashrc
RUN echo 'alias conf="python3 -m source.cli.config"' >> ~/.bashrc
RUN echo 'alias etl="python3 -m source.cli.etl"' >> ~/.bashrc
RUN echo 'alias train="python3 -m source.cli.train"' >> ~/.bashrc


### Recources
# https://blog.roboflow.com/use-the-gpu-in-docker/
# https://github.com/NVIDIA/nvidia-docker/issues/1632
# https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
