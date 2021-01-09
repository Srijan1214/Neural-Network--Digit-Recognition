#pragma once
#include <math.h>

#include <iostream>
#include <vector>
#include <unordered_map>

#include "random_engine.h"
#include "mathematical_functions.h"
#include "constants.h"


class neuron {
   private:
	std::unordered_map<neuron*, long double> m_next_layer_weights;
	std::unordered_map<neuron*, long double> m_prev_layer_weights;

	long double m_activation;
	long double m_bias;
	long double m_z_value;

	// variables for backward propagation
	long double m_del_c_by_del_a;
	std::unordered_map<neuron*, long double> m_prev_layer_gradient_weights;
	long double m_bias_gradient;

   public:
	neuron();

	void set_backward_layer_neurons(const std::vector<neuron*>& a_backward_layer);

	void set_forward_weight_value(neuron* const a_forward_neuron, const long double& a_value);
	void set_activation(const long double& a_activation);

	void compute_activation_and_z_val();
	long double get_activation() const;
	long double get_z_value() const;

	void set_del_c_by_del_a(long double x);
	long double get_del_c_by_del_a() const;

	void compute_weight_gradient_based_on_del_c_by_del_a();
	void compute_bias_gradient_based_on_del_c_by_del_a();

	void compute_del_c_by_del_a_for_hidden_layer();

	void average_gradients_and_change_weights_and_bias(const int MINI_BATCH_SIZE);
};