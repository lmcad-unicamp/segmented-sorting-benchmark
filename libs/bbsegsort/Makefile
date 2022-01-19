#ARCH=52

NVFLAGS=-std=c++14 -gencode arch=compute_${ARCH},code=sm_${ARCH} -O3 -lcuda --expt-relaxed-constexpr -DEXECUTIONS=${EXECS} -DELAPSED_TIME=${TIME}

bbsegsort.exe: main.cu 
	nvcc $(NVFLAGS) main.cu -o bbsegsort.exe

clean:
	rm -f bbsegsort.exe

.phony: clean

