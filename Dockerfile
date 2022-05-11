# Build from base tensorflow image
FROM nvidia/cuda:11.0-base

CMD nvidia-smi

# Install python 
# RUN apt-get update -y
# RUN apt install python3-pip
# RUN pip install tensorflow
# Create a working directory



# Run the init script to initialize folder structure

# Copy all source code into working directory

# Copy user defined models into source/models/<model>