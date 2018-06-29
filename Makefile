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

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart

OBJS = sim_anneal.cu.o siqadconn.cc.o main.cc.o
TARGET = exec
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)


.SUFFIXES: .cpp .cu .o

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)
