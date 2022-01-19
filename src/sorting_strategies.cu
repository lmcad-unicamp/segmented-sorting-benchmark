#include "arglib.hpp"

/* ************************************************************************************** *
 * CPU seq. sort
 * ************************************************************************************** */
int int_cmp (const void * a, const void * b) 
{
   return ( *(int*)a - *(int*)b );
}

int sort_cpu(int nsegments, int* segment_indices, int array_sz, int* array_values)
{
    for (int s = 0; s<nsegments; s++) {
	//cout << "Sort seg" << segment_indices[s] << ":" << segment_indices[s+1] << "\n";
        int* seg_start = &array_values[segment_indices[s]];
        int  seg_size  = segment_indices[s+1] - segment_indices[s];
	   	//cout << "Sort " << seg_start << " - " << seg_size << "\n";
	   	qsort(seg_start, seg_size, sizeof(int), int_cmp);
	}

    return 0; // OK
}

void print_array(int* array, int arraysz)
{
	for (int i =0; i<arraysz; i++)
		cout << ", " << array[i];
}

void print_seg_array(int* array, int arraysz, int* seg_indices, int nsegs)
{
	for (int s=0; s<nsegs; s++) {
		cout << "|";
		print_array(&array[seg_indices[s]], seg_indices[s+1]-seg_indices[s]);
		cout << "|";
	}

}


/* Return 0 if ok, -1 otherwise. */
int check_result(int* correct_result, int* array_to_check, int array_sz)
{
	// cout << "Correct:";
	// print_array(correct_result,array_sz);
	// cout << "\nTo check:";
	//print_array(array_to_check,array_sz);
	//cout << "\n";

	for(int i=0; i<array_sz; i++)
		if (correct_result[i] != array_to_check[i])
			return -1;
	return 0; // OK
}


/* ************************************************************************************** *
 * Fixthrust
 * ************************************************************************************** */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <iostream>

template <typename T, typename Op>
struct Operation {
    uint shift_val;

    Operation(uint shift_val) {
        this->shift_val = shift_val;
    }

    __host__ __device__
    T operator()(const T x, const T y)
    {
        T fix = y << shift_val;
        Op op = Op();
        return op(x, fix);
    }
};

int evaluate_fixthrust(int nsegments, int* segment_indices, int array_sz, int* array_values, int* correct_result, int nruns, std::vector<double>& time_samples, std::vector<double>* ft_pre_time_samples = NULL, std::vector<double>* ft_sort_time_samples = NULL, std::vector<double>* ft_post_time_samples = NULL)
{
    /* Copy original array. */
    thrust::host_vector<uint> h_seg_aux(nsegments + 1); // 
    for (uint i = 0; i < nsegments + 1; i++)
        h_seg_aux[i] = segment_indices[i];

    thrust::host_vector<uint> h_vec(array_sz);
    for (uint i = 0; i < array_sz; i++)
        h_vec[i] = array_values[i];

    thrust::host_vector<uint> h_seg(array_sz);
    for (uint i = 0; i < nsegments; i++) {
        for (uint j = h_seg_aux[i]; j < h_seg_aux[i + 1]; j++) {
            h_seg[j] = i;
        }
    }

    cudaEvent_t start, pre_end, sort_end, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (ft_pre_time_samples || ft_sort_time_samples || ft_post_time_samples) {
        cudaEventCreate(&pre_end);
        cudaEventCreate(&sort_end);
    }

    thrust::device_vector<uint> d_vec(array_sz);
    thrust::device_vector<uint> d_seg = h_seg;

    for (uint i = 0; i < nruns; i++) 
    {
        /* Copy array data to GPU. */
        thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

        /*
         * maximum element of the array.
         */
        cudaEventRecord(start);
        thrust::device_vector<uint>::iterator iter = thrust::max_element(d_vec.begin(), d_vec.end());
        uint max_val = *iter;
        uint mostSignificantBit = (uint)log2((double)max_val) + 1;
        /*
         * add prefix to the elements
         */
        Operation< uint, thrust::plus<uint> > op_plus(mostSignificantBit);
        thrust::transform(d_vec.begin(), d_vec.end(), d_seg.begin(), d_vec.begin(), op_plus);

        if (ft_pre_time_samples || ft_sort_time_samples || ft_post_time_samples) {
            cudaEventRecord(pre_end);
        }

        /*
         * sort the segments
         */
        thrust::sort(d_vec.begin(), d_vec.end());

        if (ft_pre_time_samples || ft_sort_time_samples || ft_post_time_samples) {
            cudaEventRecord(sort_end);
        }

        /*
         * update back the array elements
         */
        Operation< uint, thrust::minus<uint> > op_minus(mostSignificantBit);
        thrust::transform(d_vec.begin(), d_vec.end(), d_seg.begin(), d_vec.begin(), op_minus);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) {
            std::cerr << "4: Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
            continue;
        }
        if (errAsync != cudaSuccess) {
            std::cerr << "4: Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
            continue;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        time_samples.push_back(milliseconds);
        if (ft_pre_time_samples || ft_sort_time_samples || ft_post_time_samples) {
            cudaEventElapsedTime(&milliseconds, start, pre_end);
            ft_pre_time_samples->push_back(milliseconds);
            cudaEventElapsedTime(&milliseconds, pre_end, sort_end);
            ft_sort_time_samples->push_back(milliseconds);
            cudaEventElapsedTime(&milliseconds, sort_end, stop);
            ft_post_time_samples->push_back(milliseconds);
        }
    }

    /* Copy array back from GPU. */
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

    /* TODO: Check h_vec contents to verify if the code sorted the array correctly. */

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (ft_pre_time_samples || ft_sort_time_samples || ft_post_time_samples) {
        cudaEventDestroy(pre_end);
        cudaEventDestroy(sort_end);
    }

	if (correct_result)
		return check_result(correct_result,(int*) &h_vec[0],array_sz); // TODO: fix arrays to use uint
	else 
		return 0; // OK
}


/* ************************************************************************************** *
 * MergeSeg
 * ************************************************************************************** */

#include <moderngpu/kernel_segsort.hxx>
#include <moderngpu/context.hxx>

void cudaTest(cudaError_t error) 
{
    if (error != cudaSuccess) 
    {
        printf("cuda returned error %s (code %d), line(%d)\n",
                cudaGetErrorString(error), error, __LINE__);
        exit (EXIT_FAILURE);
    }
}

int evaluate_mergeseg(int nsegments, int* segment_indices, int array_sz, int* array_values, int* correct_result, int nruns, std::vector<double>& time_samples)
{
    uint mem_size_seg = sizeof(uint) * (nsegments); /* Shouldn't it be nsegments+1? */
    uint *h_seg = (uint *) malloc(mem_size_seg); /* TODO: Check malloc return value. */
    for (int i = 0; i < nsegments; i++)
        h_seg[i] = segment_indices[i];

    uint mem_size_vec = sizeof(uint) * array_sz;
    uint *h_vec = (uint *) malloc(mem_size_vec); /* TODO: Check malloc return value. */
    for (int i = 0; i < array_sz; i++)
        h_vec[i] = array_values[i];

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint *d_seg, *d_vec, *d_index_resp;

    cudaTest(cudaMalloc((void **) &d_seg, mem_size_seg));
    cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
    cudaTest(cudaMalloc((void **) &d_index_resp, mem_size_vec));

    mgpu::standard_context_t context(false); // False => avoid printing device info.

    for (uint j = 0; j < nruns; j++) 
    {
        // copy host memory to device
        cudaTest(cudaMemcpy(d_seg, h_seg, mem_size_seg, cudaMemcpyHostToDevice));
        cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));

        //try {
        cudaEventRecord(start);
        mgpu::segmented_sort(d_vec, d_index_resp, array_sz, d_seg,
                nsegments, mgpu::less_t<uint>(), context);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) {
            std::cerr << "4: Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
            continue;
        }
        if (errAsync != cudaSuccess) {
            std::cerr << "4: Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
            continue;
        }

        time_samples.push_back(milliseconds);

        context.synchronize();
    }

    cudaTest(cudaMemcpy(h_vec, d_vec, mem_size_vec, cudaMemcpyDeviceToHost));

    int result = 0; // OK
    if (correct_result)
	    result = check_result(correct_result,(int*) &h_vec[0],array_sz);

    cudaFree(d_seg);
    cudaFree(d_vec);
    cudaFree(d_index_resp);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_seg);
    free(h_vec);

	return result;
}

/* ************************************************************************************** *
 * NThurst
 * ************************************************************************************** */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

int evaluate_nthrust(int nsegments, int* segment_indices, int array_sz, int* array_values, int* correct_result, int nruns, std::vector<double>& time_samples)
{
    thrust::host_vector<int> h_seg(nsegments + 1);
    for (int i = 0; i < nsegments + 1; i++)
        h_seg[i] = segment_indices[i];

    thrust::host_vector<int> h_vec(array_sz);
    for (int i = 0; i < array_sz; i++)
        h_vec[i] = array_values[i];

    thrust::device_vector<uint> d_vec(array_sz);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (uint i = 0; i < nruns; i++) 
    {
        thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

        cudaEventRecord(start);
        for (int i = 0; i < nsegments; i++) {
            thrust::sort(d_vec.begin() + h_seg[i], d_vec.begin() + h_seg[i + 1]);
        }
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) {
            std::cerr << "4: Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
            continue;
        }
        if (errAsync != cudaSuccess) {
            std::cerr << "4: Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
            continue;
        }
        time_samples.push_back(milliseconds);
    }

    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

    /* TODO: 1) Check results (h_vec) to verify if the code sorted the array correctly. */

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	if (correct_result)
		return check_result(correct_result,&h_vec[0],array_sz);
	else 
		return 0; // OK
}

/* ************************************************************************************** *
 * RadixSeg
 * ************************************************************************************** */

#include <cub/util_allocator.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <math.h>
#include <chrono>
#include <cuda.h>

using namespace std::chrono;
using namespace cub;

int evaluate_radixseg(int nsegments, int* segment_indices, int array_sz, int* array_values, int* correct_result, int nruns, std::vector<double>& time_samples)
{
    uint mem_size_seg = sizeof(uint) * (nsegments + 1);
    int *h_seg = (int *) malloc(mem_size_seg); /* TODO: Check malloc return value. */
    for (int i = 0; i < nsegments + 1; i++)
        h_seg[i] = segment_indices[i];

    uint mem_size_vec = sizeof(uint) * array_sz;
    uint *h_vec = (uint *) malloc(mem_size_vec); /* TODO: Check malloc return value. */
    uint *h_value = (uint *) malloc(mem_size_vec); /* TODO: Check malloc return value. */
    for (int i = 0; i < array_sz; i++) {
        h_vec[i] = array_values[i];
        h_value[i] = i;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint *d_value, *d_value_out, *d_vec, *d_vec_out;
    int *d_seg;
    void *d_temp = NULL;
    size_t temp_bytes = 0;

    cudaTest(cudaMalloc((void **) &d_seg, mem_size_seg));
    cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
    cudaTest(cudaMalloc((void **) &d_value, mem_size_vec));
    cudaTest(cudaMalloc((void **) &d_vec_out, mem_size_vec));
    cudaTest(cudaMalloc((void **) &d_value_out, mem_size_vec));

    for (uint i = 0; i < nruns; i++) 
    {
        // copy host memory to device
        cudaTest(cudaMemcpy(d_seg, h_seg, mem_size_seg, cudaMemcpyHostToDevice));
        cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));
        cudaTest(cudaMemcpy(d_value, h_value, mem_size_vec, cudaMemcpyHostToDevice));

        if(temp_bytes == 0) {
            /* Invoking SortPairs to find out temp_storage_bytes (temp_bytes) and allocate d_temp_storage (d_temp). */
            cub::DeviceSegmentedRadixSort::SortPairs(d_temp, temp_bytes, d_vec,
                d_vec_out, d_value, d_value_out, array_sz, nsegments, d_seg, d_seg + 1);
            cudaTest(cudaMalloc((void **) &d_temp, temp_bytes)); 
        }
        cudaEventRecord(start);
        cub::DeviceSegmentedRadixSort::SortPairs(d_temp, temp_bytes, d_vec, d_vec_out, 
                                                 d_value, d_value_out, array_sz,
                                                 nsegments, d_seg, d_seg + 1);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) {
            std::cerr << "4: Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
            continue;
        }
        if (errAsync != cudaSuccess) {
            std::cerr << "4: Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
            continue;
        }
        time_samples.push_back(milliseconds);

        cudaDeviceSynchronize();
    }

    cudaTest(cudaMemcpy(h_value, d_value_out, mem_size_seg, cudaMemcpyDeviceToHost));
    cudaTest(cudaMemcpy(h_vec, d_vec_out, mem_size_vec, cudaMemcpyDeviceToHost));

    /* TODO: Check results (h_vec?) to verify if the code sorted the array correctly. */

    cudaFree(d_seg);
    cudaFree(d_vec);
    cudaFree(d_vec_out);
    cudaFree(d_value);
    cudaFree(d_value_out);
    cudaFree(d_temp);

    int result = 0; // OK
    if (correct_result)
	    result = check_result(correct_result,(int*) &h_vec[0],array_sz);

    free(h_seg);
    free(h_vec);
    free(h_value);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	return result;
}

/* ************************************************************************************** *
 * Fixcub
 * ************************************************************************************** */

#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <iostream>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

template <typename T>
struct Plus {
    __host__ __device__
    T operator()(const T x, const T y){ return x + y; }
};

template <typename T>
struct Minus {
    __host__ __device__
    T operator()(const T x, const T y) {return x - y;}
};

template<typename Op>
__global__ void adjustment(uint* d_vec, uint* d_seg, uint array_sz, uint* d_max ){

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < array_sz) {
        uint mostSignificantBit = (uint)log2((double)*d_max) + 1;
        uint segIndex = d_seg[id] << mostSignificantBit;
        Op op = Op();
        d_vec[id] = op(d_vec[id], segIndex);
    }
}

int evaluate_fixcub(int nsegments, int* segment_indices, int array_sz, int* array_values, int* correct_result, int nruns, std::vector<double>& time_samples, std::vector<double>* ft_pre_time_samples = NULL, std::vector<double>* ft_sort_time_samples = NULL, std::vector<double>* ft_post_time_samples = NULL)

{
    uint mem_size_seg = sizeof(uint) * (nsegments + 1);
    uint *h_seg_aux = (uint *) malloc(mem_size_seg); /* TODO: Check malloc return value. */
    for (int i = 0; i < nsegments + 1; i++)
        h_seg_aux[i] = segment_indices[i];

    int mem_size_vec = sizeof(uint) * array_sz;
    uint *h_vec = (uint *) malloc(mem_size_vec); /* TODO: Check malloc return value. */
    uint *h_value = (uint *) malloc(mem_size_vec); /* TODO: Check malloc return value. */
    for (int i = 0; i < array_sz; i++) {
        h_vec[i] = array_values[i];
        h_value[i] = i;
    }

    uint *h_seg = (uint *) malloc(mem_size_vec); /* TODO: Check malloc return value. */
    for (int i = 0; i < nsegments; i++) {
        for (uint j = h_seg_aux[i]; j < h_seg_aux[i + 1]; j++) {
            h_seg[j] = i;
        }
    }


    cudaEvent_t start, pre_end, sort_end, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (ft_pre_time_samples || ft_sort_time_samples || ft_post_time_samples) {
        cudaEventCreate(&pre_end);
        cudaEventCreate(&sort_end);
    }

    uint *d_value, *d_value_out, *d_vec, *d_vec_out, *d_max, *d_seg;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cudaTest(cudaMalloc((void **) &d_max, sizeof(uint)));
    cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
    cudaTest(cudaMalloc((void **) &d_seg, mem_size_vec));
    cudaTest(cudaMalloc((void **) &d_value, mem_size_vec));
    cudaTest(cudaMalloc((void **) &d_vec_out, mem_size_vec));
    cudaTest(cudaMalloc((void **) &d_value_out, mem_size_vec));

    cudaTest(cudaMemcpy(d_value, h_value, mem_size_vec, cudaMemcpyHostToDevice));
    cudaTest(cudaMemcpy(d_seg, h_seg, mem_size_vec, cudaMemcpyHostToDevice));

    void *d_temp = NULL;
    size_t temp_bytes = 0;
    int grid = ((array_sz-1)/BLOCK_SIZE) + 1;

    for (uint i = 0; i < nruns; i++) 
    {
        cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));

        /*
         * maximum element of the array.
         */
        cudaEventRecord(start);
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_vec, d_max, array_sz);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);    // Allocate temporary storage
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_vec, d_max, array_sz);    // Run max-reduction

        /*
         * add prefix to the elements
         */
        adjustment<Plus<uint>> <<< grid, BLOCK_SIZE>>>(d_vec, d_seg, array_sz, d_max);

        if (ft_pre_time_samples || ft_sort_time_samples || ft_post_time_samples) {
            cudaEventRecord(pre_end);
        }

        /*
         * sort the vector
         */
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_vec, d_vec_out, d_value, d_value_out, array_sz);
        cudaMalloc((void **) &d_temp, temp_bytes);
        cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_vec, d_vec_out, d_value, d_value_out, array_sz);

        if (ft_pre_time_samples || ft_sort_time_samples || ft_post_time_samples) {
            cudaEventRecord(sort_end);
        }

        adjustment<Minus<uint>> <<< grid, BLOCK_SIZE>>>(d_vec_out, d_seg, array_sz, d_max);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t errSync = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) {
            std::cerr << "4: Sync kernel error: " << cudaGetErrorString(errSync) << std::endl;
            continue;
        }
        if (errAsync != cudaSuccess) {
            std::cerr << "4: Async kernel error: " << cudaGetErrorString(errAsync) << std::endl;
            continue;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        time_samples.push_back(milliseconds);
        if (ft_pre_time_samples || ft_sort_time_samples || ft_post_time_samples) {
            cudaEventElapsedTime(&milliseconds, start, pre_end);
            ft_pre_time_samples->push_back(milliseconds);
            cudaEventElapsedTime(&milliseconds, pre_end, sort_end);
            ft_sort_time_samples->push_back(milliseconds);
            cudaEventElapsedTime(&milliseconds, sort_end, stop);
            ft_post_time_samples->push_back(milliseconds);
        }

        /* Should we de-allocate and allocate these data structures at every loop iteration? */
        cudaFree(d_temp_storage);
        temp_storage_bytes = 0;
        d_temp_storage = NULL;

        cudaFree(d_temp);
        temp_bytes = 0;
        d_temp = NULL;

        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_vec, d_vec_out, mem_size_vec, cudaMemcpyDeviceToHost);

    /* TODO: Check results (h_vec?) to verify if the code sorted the array correctly. */

    cudaFree(d_max);
    cudaFree(d_seg);
    cudaFree(d_vec);
    cudaFree(d_vec_out);
    cudaFree(d_value);
    cudaFree(d_value_out);

    int result = 0; // OK
    if (correct_result)
	    result = check_result(correct_result,(int*) &h_vec[0],array_sz);

    free(h_seg_aux);
    free(h_seg);
    free(h_vec);
    free(h_value);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (ft_pre_time_samples || ft_sort_time_samples || ft_post_time_samples) {
        cudaEventDestroy(pre_end);
        cudaEventDestroy(sort_end);
    }

	return result;
}

/* ************************************************************************************** *
 * bbsegsort
 * ************************************************************************************** */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "bbsegsort/bb_segsort.h"

#define CUDA_CHK(_e, _s) if(_e != cudaSuccess) { \
        std::cerr << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl; \
        return 1; }

template<typename T>
void print(T host_data, uint n) {
    std::cout << "\n";
    for (uint i = 0; i < n; i++) {
        std::cout << host_data[i] << " ";
    }
    std::cout << "\n";
}

int evaluate_bbsegsort(int nsegments, int* segment_indices, int array_sz, int* array_values, int* correct_result, int nruns, std::vector<double>& time_samples)
{
    std::vector<int>    h_seg(nsegments, 0);
    for (int i = 0; i < nsegments+1; i++)
        h_seg[i] = segment_indices[i];

    std::vector<int>    h_vec(array_sz, 0);
    std::vector<double> h_val(array_sz, 0.0);
    for (int i = 0; i < array_sz; i++) {
        h_vec[i] = array_values[i];
        h_val[i] = h_vec[i];
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    

    int    *key_d;
    double *val_d;
    int    *seg_d;

    cudaError_t err;
    err = cudaMalloc((void**)&key_d, sizeof(int   )*array_sz);
    CUDA_CHK(err, "alloc key_d");
    err = cudaMalloc((void**)&val_d, sizeof(double)*array_sz);
    CUDA_CHK(err, "alloc val_d");
    err = cudaMalloc((void**)&seg_d, sizeof(int   )*nsegments);
    CUDA_CHK(err, "alloc seg_d");

    err = cudaMemcpy(seg_d, &h_seg[0], sizeof(int   )*nsegments, cudaMemcpyHostToDevice);
    CUDA_CHK(err, "copy to seg_d");

    for (uint j = 0; j < nruns; j++) 
    {
        err = cudaMemcpy(key_d, &h_vec[0], sizeof(int   )*array_sz, cudaMemcpyHostToDevice);
        CUDA_CHK(err, "copy to key_d");
        err = cudaMemcpy(val_d, &h_val[0], sizeof(double)*array_sz, cudaMemcpyHostToDevice);
        CUDA_CHK(err, "copy to val_d");

        cudaEventRecord(start);
        if (bb_segsort(key_d, val_d, array_sz, seg_d, nsegments) != 0) {
            std::cerr << "Error when executing bb_segsort(...)" << std::endl;
            return 1;
        }
        cudaEventRecord(stop);

        err = cudaMemcpy(&h_vec[0], key_d, sizeof(int   )*array_sz, cudaMemcpyDeviceToHost);
        CUDA_CHK(err, "copy from key_d");
        err = cudaMemcpy(&h_val[0], val_d, sizeof(double)*array_sz, cudaMemcpyDeviceToHost);
        CUDA_CHK(err, "copy from val_d");

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        time_samples.push_back(milliseconds);

        cudaDeviceSynchronize();
    }

    /* TODO: Check results (h_vec?) (h_val?) to verify if the code sorted the array correctly. */

    err = cudaFree(key_d);
    CUDA_CHK(err, "free key_d");
    err = cudaFree(val_d);
    CUDA_CHK(err, "free val_d");
    err = cudaFree(seg_d);
    CUDA_CHK(err, "free seg_d");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	if (correct_result)
		return check_result(correct_result,&h_vec[0],array_sz);
	else 
		return 0; // OK
}

/* ************************************************************************************** *
 * Evaluate all strategies
 * ************************************************************************************** */

/* Get the median value. WARNING: This function sorts the vector datastructure. */
double get_median(std::vector<double>& v)
{
    sort(v.begin(), v.end());
    return v[(int) (v.size()/2)];
}

double get_mean(const std::vector<double>& v)
{
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

double get_stdev(const std::vector<double>& v)
{
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    std::vector<double> diff(v.size());
    std::transform(v.begin(), v.end(), diff.begin(), std::bind2nd(std::minus<double>(), mean));

    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / v.size());

    return stdev;
}

clarg::argInt nthrustMaxSegs  ("-nthrustMaxSegs", "Maximum number of segments allowed to run the N Thrust approach.", 1024);

clarg::argInt radixsegMaxSegs ("-radixsegMaxSegs", "Maximum number of segments allowed to run the Radix Seg approach.", 262144);

clarg::argBool fixthrust_flag ("-runFixThrust", "Run the FixThrust segmented sorting approach.");
clarg::argBool fix_steps_flag ("-measureFixSteps", "Measure step times when executing fix approaches.");
clarg::argBool mergeseg_flag  ("-runMergeSeg", "Run the MergeSeg segmented sorting approach.");
clarg::argBool fixcub_flag    ("-runFixCub", "Run the FixCub segmented sorting approach.");
clarg::argBool nthrust_flag   ("-runNThrust", "Run the NThrust segmented sorting approach.");
clarg::argBool radixseg_flag  ("-runRadixSeg", "Run the RadixSeg segmented sorting approach.");
clarg::argBool bbsegsort_flag ("-runBBSegSort", "Run the BBSegSort segmented sorting approach.");
clarg::argBool runAll         ("-runAll", "Run all sorting approaches.");
clarg::argInt  nRuns          ("-nRuns", "Number of times each sorting strategy must be invoked for each dataset entry.", 10);

clarg::argBool checkSortResult ("-chkResult", "Enable the check result.");


int evaluate_strategies(int nsegments, int* segment_indices, int array_sz, int* array_values, int seed) 
{
    int nruns = nRuns.get_value();

#define report_result(NAME,t_samples)                                                                            \
    std::cout << NAME << ":" << array_sz << ":" << nsegments << ":" << seed                                      \
                << ":" << get_median(t_samples)                                                                  \
                << ":" << get_mean(t_samples)                                                                    \
                << ":" << get_stdev(t_samples)                                                                   \
                << ":" << t_samples.size()                                                                       \
                << ":OK" << std::endl


#define evalstrategy(NAME,F) if (runAll.was_set() || NAME ## _flag.was_set()) {                                    \
    std::vector<double> time_samples;                                                                              \
    int    ret;                                                                                                    \
    if (ret = F(nsegments, segment_indices, array_sz, array_values, correct_result, nruns, time_samples) != 0) {   \
        std::cerr << "ERROR: " << #F << "(...) returned error" << std::endl;                                       \
        std::cout << #NAME << ":" << array_sz << ":" << nsegments << ":" << seed                                   \
                  << ":::::RUN ERROR " << ret << std::endl;                                                        \
    }                                                                                                              \
    else {                                                                                                         \
        /* Print array_sz:nsegments:seed:median:average:stdev:# of time samples */                                 \
        report_result(#NAME, time_samples);                                                                         \
    }                                                                                                              \
}

#define evalstrategy2(NAME,F) if (runAll.was_set() || NAME ## _flag.was_set()) {                                   \
    std::vector<double> time_samples;                                                                              \
    std::vector<double> pre_time_samples;                                                                          \
    std::vector<double> sort_time_samples;                                                                         \
    std::vector<double> post_time_samples;                                                                         \
    int    ret;                                                                                                    \
    if (ret = F(nsegments, segment_indices, array_sz, array_values, correct_result,                                \
                nruns, time_samples, &pre_time_samples, &sort_time_samples, &post_time_samples) != 0) {            \
        std::cerr << "ERROR: " << #F << "(...) returned error" << std::endl;                                       \
        std::cout << #NAME << ":" << array_sz << ":" << nsegments << ":" << seed                                   \
                  << ":::::RUN ERROR " << ret << std::endl;                                                        \
    }                                                                                                              \
    else {                                                                                                         \
        /* Print array_sz:nsegments:seed:median:average:stdev:# of time samples */                                 \
        report_result(#NAME, time_samples);                                                                        \
        report_result(#NAME "-pre", pre_time_samples);                                                             \
        report_result(#NAME "-sort", sort_time_samples);                                                           \
        report_result(#NAME "-post", post_time_samples);                                                           \
    }                                                                                                              \
}

	int* correct_result = NULL;
	if (checkSortResult.was_set()) {
		correct_result = (int*) malloc(sizeof(int)*array_sz);
		memcpy(correct_result,array_values,sizeof(int)*array_sz);
		//print_seg_array(correct_result,array_sz,segment_indices,nsegments);
		sort_cpu(nsegments,segment_indices,array_sz,correct_result);
	        //print_seg_array(correct_result,array_sz,segment_indices,nsegments); cout << "\n";
	
	}

    if (fix_steps_flag.was_set()) {
      evalstrategy2(fixthrust,evaluate_fixthrust)
    }
    else {
      evalstrategy(fixthrust,evaluate_fixthrust)
    }

    evalstrategy(mergeseg,evaluate_mergeseg)

    if (fix_steps_flag.was_set()) {
      evalstrategy2(fixcub,evaluate_fixcub)
    }
    else {
      evalstrategy(fixcub,evaluate_fixcub)
    }

    if (nthrustMaxSegs.get_value() >= nsegments)
        evalstrategy(nthrust,evaluate_nthrust)
    
    if (radixsegMaxSegs.get_value() >= nsegments)
        evalstrategy(radixseg,evaluate_radixseg)
    
    evalstrategy(bbsegsort,evaluate_bbsegsort)

	if (correct_result) {
		free(correct_result);
	}

    return 0; // OK
}

