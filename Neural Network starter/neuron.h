#pragma once
#include <iostream>
#include <vector>
#include <math.h>

class neuron {
	//static std::vector<neuron*> infront_neuron_layer;
	//static std::vector<neuron*> behind_neuron_layer;
	//std::vector<double> weights;
private:
	double m_activation;
	double m_bias;
	double z;
	
	double sigmoid(double input);
public:
	neuron(std::vector<double> &,std::vector<double> &);
	void reinitialize_activation(std::vector<double> &,std::vector<double> &);
	double get_activation_value();
	double get_z();
	void change_bias(double);
	void set_bias(double);
};