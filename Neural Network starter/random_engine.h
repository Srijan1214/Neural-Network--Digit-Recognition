#pragma once
#include <random>

class random_engine {
   private:
	const static int UPPER_BOUND = 3;
	const static int LOWER_BOUND = -3;

	static std::normal_distribution<long double> m_normal_distribution_obj;
	static std::default_random_engine m_re;

   public:

	static void seed();
	static long double get_random_double();
};