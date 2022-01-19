


deviceQuerySrc=${CUDA_HOME}/samples/1_Utilities/deviceQuery/deviceQuery.cpp
deviceQueryInc=${CUDA_HOME}/samples/common/inc

BINARIES=deviceQuery.x

all: $(BINARIES)

deviceQuery.x: $(deviceQuerySrc)
	nvcc -ccbin g++ -I$(deviceQueryInc) -m64 -gencode arch=compute_${ARCH},code=sm_${ARCH} -o deviceQuery.x $(deviceQuerySrc) 

clean:
	rm -f $(BINARIES)
