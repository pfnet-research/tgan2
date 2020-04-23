FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    gfortran \
    git \
    graphviz \
    ibverbs-utils \
    imagemagick \
    infiniband-diags \
    libibumad-dev \
    libibverbs-dev \
    libmlx4-1 \
    libz-dev \
    openssh-client \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    unrar \
    unzip \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Install OpenMPI with cuda & verbs
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
RUN cd /usr/local/lib && \
    curl -L -O https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.1.tar.gz && \
    tar zxvf openmpi-3.1.1.tar.gz && rm -rf openmpi-3.1.1.tar.gz && \
    cd openmpi-3.1.1 && \
    ./configure --with-cuda --with-verbs && \
    make -j8 && make install && \
    ompi_info --parsable --all | grep -q "mpi_built_with_cuda_support:value:true"

ARG USER_ID=1000
ARG USER_NAME=ubuntu
ARG PASSWORD=ubuntu

RUN mkdir -p /home/${USER_NAME}
RUN useradd -d /home/${USER_NAME} -G sudo -u ${USER_ID} ${USER_NAME} -p $(openssl passwd -1 ${PASSWORD})
RUN chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}

USER ${USER_NAME}
ENV HOME /home/${USER_NAME}

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/lib
ENV LIBRARY_PATH=$LIBRARY_PATH:$HOME/lib
ENV CPATH=$CPATH:/usr/local/include:$HOME/include
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:$HOME/lib' >> ~/.bashrc && \
    echo 'export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib:$HOME/lib' >> ~/.bashrc && \
    echo 'export CPATH=$CPATH:/usr/local/include:$HOME/include' >> ~/.bashrc

# Install Python packages
USER root
RUN pip3 install numpy
RUN pip3 install \
    Cython==0.29.16 \
    chainer==6.0.0 \
    chainercv==0.13.1 \
    cupy-cuda100==6.0.0 \
    h5py==2.10.0 \
    imageio==2.8.0 \
    imageio-ffmpeg==0.4.1 \
    ipdb==0.13.2 \
    jupyter==1.0.0 \
    matplotlib==3.2.1 \
    mpi4py==3.0.3 \
    opencv-python==4.2.0.32 \
    pandas==0.25.3 \
    protobuf==3.11.3 \
    pytest==5.4.1 \
    PyYAML==5.3.1 \
    scipy==1.4.1 \
    scikit-image==0.16.2 \
    scikit-learn==0.22.2.post1 \
    tqdm==4.44.1 \
    xlrd==1.2.0

USER root
RUN chown ${USER_NAME}:${USER_NAME} /home/${USER_NAME}
USER ${USER_NAME}
ENV HOME /home/${USER_NAME}

