#include "array_generator_funcs.h"
#include <float.h> // DBLMAX
#include <iostream>

std::uniform_real_distribution<> distr(std::nextafter(0.0, DBL_MAX), 1.0);

double UniformRandomNumberGenerator::genNext() 
{
	return distr(m_generator);
}

double PowerLawRandomNumberGenerator::genNext() 
{
	/* TODO: Rafael, why parameter x0 == 0.1? Shouldn't it be 0.0? */
	return powerlaw(distr(m_generator), 0.1, 1.0, m_alpha);
}

double PowerLawRandomNumberGenerator::powerlaw(double y, double x0, double x1, double n)
{
	return pow(((pow(x1,(n+1)) - pow(x0,(n+1)))*y + pow(x0,(n+1))),(1/(n+1)));
}

int segments_gen(RandomNumberGenerator& random_num_generator, int array_sz, int nsegs, int* segments_indices)
{
	/* Allocate auxiliary array. */
	double* aux = new double[nsegs+1];
	if (aux == 0) return -1;

	/* Step 1: Generate random segment sizes using a power-law distribution */
	double total = 0;
	for (int i = 0; i < nsegs; i++) {
		aux[i] = random_num_generator.genNext();
		total += aux[i];
	}

	/* Step 2: Normalize segment sizes so their sum = array_sz. */
	int sum = 0;
	for (int i = 0; i < nsegs; i++) {
		/* Normalize the segment size. */
		segments_indices[i] = (int) ((aux[i] * (double) array_sz) / total);
		if(segments_indices[i] == 0)
			/* Fix 1: Ensure all segments have at least one element. */
			segments_indices[i] = 1;
		sum += segments_indices[i];
	}

	delete[] aux;

    /* Step 3: Adjust the size of the segments so the total 
	   number of elements = array_sz*/
	int remaining = array_sz - sum;

	if (remaining > 0) 
	{
		/* Distribute the remaining elements - This may be necessary 
		   due to the rounding performed in step 2. */
    	int i = 0;
		while(remaining > 0) 
		{
			segments_indices[i] += 1;
			remaining -= 1;
			i = (i+1) % nsegs;
		}
	}
	else if (remaining < 0) {
		/* Remove excess that may be caused by Fix 1. */
		int i = 0;
		while(remaining < 0) 
		{
			if(segments_indices[i] > 1) {
				/* If the segment has at least two elements, remove one. */
				segments_indices[i] -= 1;
				remaining += 1;
			}
			/* Go to the next segment. */
			i = (i+1) % nsegs;
		}
	}

	/* Step 4: Convert segment sizes into segment start indices. */
	int next_seg_start = 0;
	for (int i = 0; i < nsegs; i++) {
		int current_seg_start = next_seg_start;
		next_seg_start += segments_indices[i];
		segments_indices[i] = current_seg_start;
	}
	segments_indices[nsegs] = array_sz;

	return 0; // OK
}

int int_array_gen(RandomNumberGenerator& random_num_generator, int array_sz, int max_number_of_bits, int* array)
{
	double max_value = pow(2, max_number_of_bits);

	for (int i = 0; i < array_sz; i++)
	{
		double n = random_num_generator.genNext();
		array[i] = (int) (n * max_value);
	}
	return 0; // OK
}

