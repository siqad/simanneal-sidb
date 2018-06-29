CUDA_INSTALL_PATH ?= /usr/lib/cuda

CXX := g++
LINK := g++ -fPIC
NVCC  := nvcc -ccbin /usr/bin

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Common flags
COMMONFLAGS += $(INCLUDES) -pthread
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -pthread

OBJS = sim_anneal.cu.o siqadconn.cc.o main.cc.o
TARGET = simanneal
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)


.SUFFIXES: .cc .cu .o

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cc.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)
