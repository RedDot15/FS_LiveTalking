# Use NVIDIA's CUDA base image

# This line is commented out; it refers to a CUDA 11.3 runtime image based on Ubuntu 18.04.
    # FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu18.04 

# This line is commented out; it refers to a specific custom CUDA 11.3 development image with JupyterLab.
    # FROM m11007322/cuda11.3.0-cudnn8-devel-ubuntu20.04-jupyterlab 
    
# Specifies the base Docker image: NVIDIA CUDA 11.3.1 development image with cuDNN 8 on Ubuntu 20.04. This image includes CUDA toolkit and development libraries.
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 


# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter
RUN pip3 install --no-cache-dir jupyter

# Install basic tools
RUN apt-get update -yq --fix-missing \
 && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    pkg-config \
    wget \
    cmake \
    curl \
    git \
    vim

# Create a new Conda environment
RUN apt-get update && apt-get install -y wget \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh \
    && apt-get remove --purge --auto-remove -y wget \
    && apt-get clean \
    && ln -s /opt/conda/bin/conda /usr/bin/conda \
    && conda update -n base -c defaults conda 

# Sets the default shell for subsequent RUN commands to bash in interactive mode, which is necessary for sourcing .bashrc.
SHELL ["/bin/bash","-ic"]

## Add CUDA global variables
# Appends CUDA_HOME environment variable definition to .bashrc.
RUN echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc \ 
    # Appends CUDA's binary directory to the PATH in .bashrc.
    && echo "export PATH=${CUDA_HOME}/bin:$PATH" >> ~/.bashrc \
    # Appends CUDA's library directory to LD_LIBRARY_PATH in .bashrc.
    && echo "export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc \
    # Sources the .bashrc file to apply the new environment variables immediately within this RUN command's shell session.
    && source ~/.bashrc
    
    # This line is commented out; it would add Google's DNS server to resolv.conf, potentially for network resolution issues.
        #&& echo "nameserver 8.8.8.8" >> /etc/resolv.conf

## Install cv2 dependencies, fix libGL.so.1 error
# Updates the package list again.
RUN apt-get update
# Installs ffmpeg and libraries (libsm6, libxext6) commonly required by OpenCV (cv2) to fix potential display-related errors like 'libGL.so.1 not found'.
RUN apt-get install ffmpeg libsm6 libxext6  -y

## Configure Jupyter
# Sets an environment variable to enable JupyterLab when Jupyter starts.
ENV JUPYTER_ENABLE_LAB=yes 
# Sets the 'USER' environment variable to 'root'.
ENV USER=root
# Sets the 'HOME' environment variable to the home directory of the 'root' user.
ENV HOME=/home/$USER

## Set working directory
# Sets the working directory for any subsequent instructions in the Dockerfile, and also for the container's runtime.
WORKDIR /root 

## Set startup command
# Defines the command that will be executed when the container starts: launches JupyterLab, making it accessible from any IP (*), on port 8888, without automatically opening a browser, and allowing it to run as the root user.
CMD ["jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]