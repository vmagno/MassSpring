
###############
# CUDA stuff
CUDA_PATH ?= /opt/cuda

GPU_ARCH ?= compute_30
GPU_CODE ?= sm_52

NVCC := nvcc
NVCCFLAGS += --gpu-architecture $(GPU_ARCH) --gpu-code $(GPU_CODE)
#NVCCFLAGS += --compiler-options
#NVCCFLAGS += -v

CUDA_LIBS := -L$(CUDA_PATH)/lib64 -lcudart

###############

CXX := clang++

CXXFLAGS += -g -W -Wall
CXXFLAGS += -Wno-unused-parameter -Wno-deprecated-declarations
CXXFLAGS += $(shell pkg-config --cflags glew)
CXXFLAGS += $(shell pkg-config --cflags sdl2)
CXXFLAGS += -Wno-unused-private-field -std=c++11

INCLUDES += -I$(CUDA_PATH)/include
CXXFLAGS += $(INCLUDES)

LDFLAGS += -g
LDFLAGS += $(shell pkg-config --libs glew)
LDFLAGS += $(shell pkg-config --libs sdl2)
LDFLAGS += $(CUDA_LIBS)

TARGET := MassSpring

SRC := $(shell find . -name '*.cpp')
DEPFILES := $(patsubst %.cpp,%.d,$(SRC))
OBJ := $(patsubst %.cpp,%.o,$(SRC))

CUSRC := $(shell find . -name '*.cu')
CUOBJ := $(patsubst %.cu,%.o,$(CUSRC))
CUDEPS := $(patsubst %.cu,%.d,$(CUSRC))

.SUFFIXES: .cpp .h .o .d .cu .cuh
.PHONY: all run clean clean_all info deps


NODEPS := info clean clean_all deps



all : $(TARGET)

run : $(TARGET)
	./$(TARGET)

$(TARGET) : $(OBJ) $(CUOBJ)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ) $(CUOBJ)

deps: $(DEPFILES)

info:
	@echo "TARGET = $(TARGET)"
	@echo "SRC = $(SRC)"
	@echo "DEPFILES = $(DEPFILES)"
	@echo "OBJ = $(OBJ)"
	@echo
	@echo "CXX = $(CXX)"
	@echo "CXXFLAGS = $(CXXFLAGS)"
	@echo "LDFLAGS = $(LDFLAGS)"

%.o: %.cpp %.h %.d
	@echo "My rule"
	$(CXX) $(CXXFLAGS) -o $@ -c $<

%.o: %.cpp %.d
	@echo "My rule 2"
	$(CXX) $(CXXFLAGS) -o $@ -c $<

%.d: %.cpp
	@echo "Depfile for $<"
	$(CXX) $(CXXFLAGS) -MM -MF $@ -c $<

%.o: %.cu %.d
	@echo "CUDA file"
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.d: %.cu
	@echo "Depfile for $<"
	$(NVCC) $(NVCCFLAGS) -M -o $@ $<

clean :
	rm -rf $(OBJ) $(CUOBJ) $(TARGET)

clean_all: clean
	rm -rf $(DEPFILES) $(CUDEPS)

#Don't create dependencies when we're cleaning, for instance
#ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NODEPS))))
    #Chances are, these files don't exist.  GMake will create them and
    #clean up automatically afterwards
    -include $(DEPFILES)
    -include $(CUDEPS)
#endif
