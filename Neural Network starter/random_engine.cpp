#include "random_engine.h"

std::normal_distribution<long double> random_engine::m_normal_distribution_obj(0.0, 1.0);
std::default_random_engine random_engine::m_re;

long double random_engine::get_random_double() {
	return m_normal_distribution_obj(m_re);
}

void random_engine::seed() {
	m_re.seed(time(0));
}
