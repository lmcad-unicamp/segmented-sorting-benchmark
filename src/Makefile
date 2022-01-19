#NOTE: I had to add -DTHRUST_IGNORE_CUB_VERSION_CHECK to work with CUDA 11.3
MERGESEG_FLAGS=--expt-extended-lambda -lcuda
BBSEGSORT_FLAGS=--expt-relaxed-constexpr
NVCCFLAGS= -gencode arch=compute_${ARCH},code=sm_${ARCH} -O3 -DTHRUST_IGNORE_CUB_VERSION_CHECK -std=c++14
CCFLAGS= -O3 -std=c++14

SRC_FILES=arglib.cpp array_generator_funcs.cpp segsort-benchmark.cpp sorting_strategies.cu
HEADERS=arglib.hpp array_generator_funcs.h
LIBDIR=-I../libs

BINARIES=segort-benchmark.x

segsort-benchmark.x: $(SRC_FILES) $(HEADERS)
	nvcc $(NVCCFLAGS) $(LIBDIR) $(MERGESEG_FLAGS) $(BBSEGSORT_FLAGS) $(SRC_FILES) -o segsort-benchmark.x

# Dry-run (DR) binary
DR_SRC_FILES=arglib.cpp array_generator_funcs.cpp segsort-benchmark.cpp sorting_strategies-dryrun.cpp
DR_HEADERS=arglib.hpp array_generator_funcs.h
DR_LIBDIR=-I./

segsort-benchmark-dr.x: $(DR_SRC_FILES) $(DR_HEADERS)
	g++ $(CCFLAGS) $(DR_LIBDIR) $(DR_SRC_FILES) -o segsort-benchmark-dr.x

clean:
	rm -f $(BINARIES)

.phony: clean
