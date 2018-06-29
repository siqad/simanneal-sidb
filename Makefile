# Note: Windows OS-specific implementation not set. This means that PKG_CONFIG ?= pkg-config is ommitted.

CUDA_INSTALL_PATH ?= /usr/lib/cuda

# Compilers
CXX := g++
LINK := g++ -fPIC
NVCC  := nvcc -ccbin /usr/bin

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS := $(COMMONFLAGS) -gencode arch=compute_30,code=sm_30
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

# Compiler flags
DLINKFLAGS      := -lcublas_device -lcudadevrt -lcublas -lcudart -pthread

# Debug mode
NVCCFLAGS += --compiler-options -Wall -G

#-arch=sm_35

# Libraries
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

# Folder structure
OBJ = obj
SRC = src
INC = include

# Options
OBJS_CU = $(OBJ)/sim_anneal.h.o $(OBJ)/main.cc.o
OBJS = $(OBJ)/siqadconn.cc.o $(OBJ)/link.o
TARGET = simanneal
LINKLINE = $(LINK) -o $(TARGET) $(OBJS_CU) $(OBJS) $(LIB_CUDA)

.SUFFIXES: .c .cc .h .cu .cuh .o

$(OBJ)/%.cuh.o: $(SRC)/%.cu $(INC)/%.cuh
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(OBJ)/%.ch.o: $(SRC)/%.cu $(INC)/%.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(OBJ)/%.cu.o: $(SRC)/%.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(OBJ)/link.o:
	$(NVCC) $(NVCCFLAGS) -dlink $(OBJS_CU) -o $@

$(OBJ)/%.cc.o: $(SRC)/%.cc $(INC)/%.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS_CU) $(OBJS) Makefile
	$(LINKLINE)

clean:
	rm -rf *.o log.txt
