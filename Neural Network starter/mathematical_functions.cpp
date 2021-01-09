#include "mathematical_functions.h"

long double mathematical_functions::sigmoid(long double input) {
	return 1/(1+exp(-input));
}

long double mathematical_functions::sigmoid_prime(long double input) {
	return (sigmoid(input)*(1-sigmoid(input)));
}