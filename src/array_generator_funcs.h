#include <random>

#ifndef ARRAY_GENERATOR_FUNC_H
#define ARRAY_GENERATOR_FUNC_H

class RandomNumberGenerator
{
	public:
		virtual double genNext() = 0;
};

class PowerLawRandomNumberGenerator : public RandomNumberGenerator
{
	public:

		PowerLawRandomNumberGenerator(int seed, double alpha) : 
			m_seed(seed), m_alpha(alpha), m_generator(seed) {}

		double genNext();

	private:

		double powerlaw(double y, double x0, double x1, double n);

		int m_seed;
		double m_alpha;
		std::mt19937 m_generator;
};

class UniformRandomNumberGenerator : public RandomNumberGenerator
{
	public:

		UniformRandomNumberGenerator(int seed) : 
			m_seed(seed), m_generator(seed) {}

		double genNext();

	private:

		int m_seed;
		std::mt19937 m_generator;
};


class ConstantNumberGenerator : public RandomNumberGenerator
{
	public:

		ConstantNumberGenerator(int constantnumber) : 
			m_number(constantnumber) {}

		double genNext() { return m_number; }

	private:

		int m_number;
};

/* 
 * Fills an array with nsegs+1 integers with values indicating the start/end of each segment. 
 * The ith item contains an integer value that indicates the start index of the ith segment. 
 * The last element (nsegs+1) contains the value array_sz.
 * Returns 0 if OK, != otherwise.
 */
int segments_gen(RandomNumberGenerator& random_num_generator, int array_sz, int nsegs, int* segments_indices);

/* 
 * Fills an array with array_sz integer elements with random values ranging from 0 to 2^max_number_of_bits. 
 * Returns 0 if OK, != otherwise.
 */
int int_array_gen(RandomNumberGenerator& random_num_generator, int array_sz, int max_number_of_bits, int* array_values);

#endif  // ARRAY_GENERATOR_FUNC_H

