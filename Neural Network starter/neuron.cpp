#include "neuron.h"

neuron::neuron() : m_bias(random_engine::get_random_double()), m_bias_gradient(0) {}

void neuron::set_backward_layer_neurons(const std::vector<neuron*>& a_backward_layer) {
	for(auto prev_neuron:a_backward_layer) {
		long double weight = random_engine::get_random_double();
		m_prev_layer_weights[prev_neuron] = weight;
		prev_neuron->set_forward_weight_value(this, weight);
	}
}

void neuron::set_forward_weight_value(neuron* const a_forward_neuron, const long double& a_value) {
	m_next_layer_weights[a_forward_neuron] = a_value;
}

void neuron::set_activation(const long double& a_activation) {
	m_activation = a_activation;
}

void neuron::compute_activation_and_z_val() {
	long double z_val = 0;
	for(auto& prev_layer_weights_pair: m_prev_layer_weights) {
		neuron* const &prev_layer_neuron =prev_layer_weights_pair.first;
		long double& weight = prev_layer_weights_pair.second;
		z_val+= weight * prev_layer_neuron->get_activation();
	}
	z_val+= m_bias;

	m_z_value = z_val;
	m_activation = mathematical_functions::sigmoid(m_z_value);
}

long double neuron::get_activation() const {
	return m_activation;
}

long double neuron::get_z_value() const {
	return m_z_value;
}

void neuron::set_del_c_by_del_a(long double x) {
	m_del_c_by_del_a = x;
}

long double neuron::get_del_c_by_del_a() const {
	return m_del_c_by_del_a;
}

void neuron::compute_weight_gradient_based_on_del_c_by_del_a() {
	long double del_c_by_del_a = get_del_c_by_del_a();
	long double del_a_by_del_z = mathematical_functions::sigmoid_prime(m_z_value);
	for(auto& prev_layer_gradient_weights_pair: m_prev_layer_gradient_weights) {
		neuron* const &prev_layer_neuron =prev_layer_gradient_weights_pair.first;
		long double& weight_gradient = prev_layer_gradient_weights_pair.second;

		long double del_z_by_del_w = prev_layer_neuron->get_activation();
		long double del_c_by_del_w = del_c_by_del_a * del_a_by_del_z * del_z_by_del_w;
		weight_gradient+= del_c_by_del_w;
	}
}

void neuron::compute_bias_gradient_based_on_del_c_by_del_a() {
	long double del_c_by_del_a = get_del_c_by_del_a();
	long double del_a_by_del_z = mathematical_functions::sigmoid_prime(m_z_value);
	long double del_c_by_del_b = del_c_by_del_a * del_a_by_del_z;
	m_bias_gradient+=del_c_by_del_b;
}

void neuron::compute_del_c_by_del_a_for_hidden_layer() {
	m_del_c_by_del_a = 0;
	for(const auto& ele: m_next_layer_weights) {
		neuron* next_neuron = ele.first;
		long double del_c_by_del_a_next = next_neuron->get_del_c_by_del_a();
		long double del_a_by_del_z = mathematical_functions::sigmoid_prime(next_neuron->get_z_value());
		long double del_z_by_del_a = m_next_layer_weights[next_neuron];
		m_del_c_by_del_a += (del_c_by_del_a_next * del_a_by_del_z * del_z_by_del_a);
	}
}

void neuron::average_gradients_and_change_weights_and_bias(const int MINI_BATCH_SIZE) {
    // change weights
	for(auto& prev_layer_weights_pair: m_prev_layer_weights) {
		neuron* const &prev_layer_neuron =prev_layer_weights_pair.first;
		long double& weight = prev_layer_weights_pair.second;

		weight -= (Constants::LEARNING_RATE * m_prev_layer_gradient_weights[prev_layer_neuron]/MINI_BATCH_SIZE);
		prev_layer_neuron->set_forward_weight_value(this, weight);
		m_prev_layer_gradient_weights[prev_layer_neuron] = 0;
	}

	// change bias
	m_bias -= (Constants::LEARNING_RATE * m_bias_gradient/ MINI_BATCH_SIZE);
	m_bias_gradient = 0;
}
