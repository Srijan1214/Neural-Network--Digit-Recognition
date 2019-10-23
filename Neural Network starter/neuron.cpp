#include "neuron.h"
#include <iostream>

double neuron::sigmoid(double input) {
	return 1 / (1 + exp(-input));
}

neuron::neuron(std::vector<double> &weights,std::vector<double> &activations) :
m_activation(0),m_bias(0),z(0)
{
	reinitialize_activation(weights, activations);
	m_bias=0;
}

void neuron::reinitialize_activation(std::vector<double> &weights, std::vector<double> &activations) {
	z = 0;
	for (size_t i = 0; i < weights.size(); i++) {
		z += activations[i] * weights[i];
	}
	z += m_bias;
	//std::cout<<"z="<<z<<std::endl;
	m_activation = sigmoid(z);
	//std::cout<<"activaton="<<m_activation<<std::endl;
}

double neuron::get_activation_value(){
	return m_activation;
}

double neuron::get_z(){
	return z;
}

void neuron::change_bias(double gradient){
	m_bias-=gradient;
	//std::cout<<m_bias<<"bias"<<std::endl;
}

void neuron::set_bias(double input){
	m_bias=input;
}
