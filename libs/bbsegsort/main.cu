/*
* (c) 2015 Virginia Polytechnic Institute & State University (Virginia Tech)   
*                                                                              
*   This program is free software: you can redistribute it and/or modify       
*   it under the terms of the GNU General Public License as published by       
*   the Free Software Foundation, version 2.1                                  
*                                                                              
*   This program is distributed in the hope that it will be useful,            
*   but WITHOUT ANY WARRANTY; without even the implied warranty of             
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              
*   GNU General Public License, version 2.1, for more details.                 
*                                                                              
*   You should have received a copy of the GNU General Public License          
*                                                                              
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#include "bb_segsort.h"

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

#ifndef EXECUTIONS
#define EXECUTIONS 1
#endif

using namespace std;

#define CUDA_CHK(_e, _s) if(_e != cudaSuccess) { \
        std::cerr << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl; \
        return 1; }

template<class K, class T>
void gold_segsort(vector<K> &key, vector<T> &val, int n, const vector<int> &seg, int m);

int show_mem_usage();

template<typename T>
void print(T host_data, uint n) {
	std::cout << "\n";
	for (uint i = 0; i < n; i++) {
		std::cout << host_data[i] << " ";
	}
	std::cout << "\n";
}

int main(int argc, char **argv)
{
	uint num_of_segments;
	uint num_of_elements;
	uint i;

	scanf("%d", &num_of_segments);

	vector<int>    h_seg(num_of_segments, 0);
	for (i = 0; i < num_of_segments+1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);

	vector<int>    h_vec(num_of_elements, 0);
	vector<double> h_val(num_of_elements, 0.0);
	for (i = 0; i < num_of_elements; i++) {
		scanf("%d", &h_vec[i]);
		h_val[i] = h_vec[i];
	}


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int    *key_d;
	double *val_d;
	int    *seg_d;
	cudaError_t err;

	err = cudaMalloc((void**)&key_d, sizeof(int   )*num_of_elements);
	CUDA_CHK(err, "alloc key_d");
	err = cudaMalloc((void**)&val_d, sizeof(double)*num_of_elements);
	CUDA_CHK(err, "alloc val_d");
	err = cudaMalloc((void**)&seg_d, sizeof(int   )*num_of_segments);
	CUDA_CHK(err, "alloc seg_d");

	err = cudaMemcpy(seg_d, &h_seg[0], sizeof(int   )*num_of_segments, cudaMemcpyHostToDevice);
	CUDA_CHK(err, "copy to seg_d");

	float averageExecutions = 0;
	for (uint j = 0; j < EXECUTIONS; j++) {
		err = cudaMemcpy(key_d, &h_vec[0], sizeof(int   )*num_of_elements, cudaMemcpyHostToDevice);
		CUDA_CHK(err, "copy to key_d");
		err = cudaMemcpy(val_d, &h_val[0], sizeof(double)*num_of_elements, cudaMemcpyHostToDevice);
		CUDA_CHK(err, "copy to val_d");

		cudaEventRecord(start);
		if (bb_segsort(key_d, val_d, num_of_elements, seg_d, num_of_segments) != 0) {
			std::cerr << "Error when executing bb_segsort(...)" << std::endl;
			return 1;
		}
		cudaEventRecord(stop);

		err = cudaMemcpy(&h_vec[0], key_d, sizeof(int   )*num_of_elements, cudaMemcpyDeviceToHost);
		CUDA_CHK(err, "copy from key_d");
		err = cudaMemcpy(&h_val[0], val_d, sizeof(double)*num_of_elements, cudaMemcpyDeviceToHost);
		CUDA_CHK(err, "copy from val_d");

		if (ELAPSED_TIME == 1) {
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			averageExecutions += milliseconds;
			//std::cout << milliseconds << "\n";
		}

		cudaDeviceSynchronize();
	}

    err = cudaFree(key_d);
    CUDA_CHK(err, "free key_d");
    err = cudaFree(val_d);
    CUDA_CHK(err, "free val_d");
    err = cudaFree(seg_d);
    CUDA_CHK(err, "free seg_d");

	if (ELAPSED_TIME != 1) {
		print(h_vec.data(), num_of_elements);
	}
	else {	std::cout << averageExecutions/EXECUTIONS << "\n";}

	return 0;

}

/*show_mem_usage();
gold_segsort(h_vec, h_val, num_of_elements, h_seg, num_of_segments);
int cnt = 0;
for(int i = 0; i < num_of_elements; i++)
	if(h_vec[i] != key_h[i]) cnt++;
if(cnt != 0) printf("[NOT PASSED] checking keys: #err = %i (%4.2f%% #nnz)\n", cnt, 100.0*(double)cnt/num_of_elements);
else printf("[PASSED] checking keys\n");
cnt = 0;
for(int i = 0; i < num_of_elements; i++)
	if(h_val[i] != val_h[i]) cnt++;
if(cnt != 0) printf("[NOT PASSED] checking vals: #err = %i (%4.2f%% #nnz)\n", cnt, 100.0*(double)cnt/num_of_elements);
else printf("[PASSED] checking vals\n");*/

template<class K, class T>
void gold_segsort(vector<K> &key, vector<T> &val, int n, const vector<int> &seg, int m)
{
    vector<pair<K,T>> pairs;
    for(int i = 0; i < n; i++)
    {
        pairs.push_back({key[i], val[i]});
    }

    for(int i = 0; i < m; i++)
    {
        int st = seg[i];
        int ed = (i<m-1)?seg[i+1]:n;
        stable_sort(pairs.begin()+st, pairs.begin()+ed, [&](pair<K,T> a, pair<K,T> b){ return a.first < b.first;});
        // sort(pairs.begin()+st, pairs.begin()+ed, [&](pair<K,T> a, pair<K,T> b){ return a.first < b.first;});
    }
    
    for(int i = 0; i < n; i++)
    {
        key[i] = pairs[i].first;
        val[i] = pairs[i].second;
    }
}

int show_mem_usage()
{
    cudaError_t err;
     // show memory usage of GPU
    size_t free_byte ;
    size_t total_byte ;
    err = cudaMemGetInfo(&free_byte, &total_byte);
    CUDA_CHK(err, "check memory info.");
    size_t used_byte  = total_byte - free_byte;
    printf("GPU memory usage: used = %4.2lf MB, free = %4.2lf MB, total = %4.2lf MB\n", 
        used_byte/1024.0/1024.0, free_byte/1024.0/1024.0, total_byte/1024.0/1024.0);   
    return cudaSuccess;
}
