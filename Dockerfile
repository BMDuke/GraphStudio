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

# Run the init script to initialize folder structure
COPY app/source/setup_project.py /graph-studio/setup.py
CMD ["python3", "setup.py"]

# Copy all source code into working directory

# Copy user defined models into source/models/<model>


### Recources
# https://blog.roboflow.com/use-the-gpu-in-docker/
# https://github.com/NVIDIA/nvidia-docker/issues/1632
# https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
