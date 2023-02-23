FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /app

# simanneal packages
RUN apt-get -y update && apt-get install -y \
    cmake \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-random-dev

# extra Nvidia packages
RUN apt-get install -y \
    cuda-nsight-systems-11-8 \
    cuda-nsight-compute-11-8 \
    nsight-systems-2022.4.2

#RUN nvcc src/simanneal_cuda.cu -o add_cuda
#RUN nvprof ./add_cuda
