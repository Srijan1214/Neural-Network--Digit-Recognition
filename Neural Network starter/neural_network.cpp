#include "neural_network.h"

neural_network::neural_network(int a_input_dimension,
							   int a_no_of_classifications)
	: m_neuron_layers(Constants::NUMBER_OF_HIDDEN_LAYERS + 2) {

	random_engine::seed();

	m_neuron_layers.front() = std::vector<neuron *>(a_input_dimension);
	for(auto& ele:m_neuron_layers.front()){
		ele = new neuron();
	}

	m_neuron_layers.back() = std::vector<neuron *>(a_no_of_classifications);
	for(auto& ele:m_neuron_layers.back()){
		ele = new neuron();
	}

	for(int i = 1; i < m_neuron_layers.size() - 1; i++) {
		m_neuron_layers[i] = std::vector<neuron *>(Constants::NEURONS_PER_HIDDEN_LAYER);
		for(auto& ele:m_neuron_layers[i]){
			ele = new neuron();
		}
	}

	for (int i = m_neuron_layers.size() - 1; i > 0; i--) {
		for (int j = 0; j < m_neuron_layers[i].size(); j++) {
			m_neuron_layers[i][j]->set_backward_layer_neurons(m_neuron_layers[i - 1]);
		}
	}
}

void neural_network::propagate_forward(const std::vector<long double> &a_input) const {
    assert(a_input.size() == m_neuron_layers.front().size());

	// Set the Input Layer
	for (int i = 0; i < m_neuron_layers.front().size(); i++) {
		m_neuron_layers.front()[i]->set_activation(a_input[i]/255);
	}
	
	for (int i = 1; i < m_neuron_layers.size(); i++) {
		for(int j = 0; j < m_neuron_layers[i].size(); j++) {
			m_neuron_layers[i][j]->compute_activation_and_z_val();
		}
	}
}

void neural_network::perform_SGD_training(std::vector<std::vector<long double>>& train_inputs, std::vector<int>& train_actual_outputs) {
	// Use shuffled indexes as the input for the training
	std::vector<int> shuffled_indexes;
	for (int i = 0; i < train_actual_outputs.size(); ++i) {
		shuffled_indexes.push_back(i);
	}
	std::random_shuffle(shuffled_indexes.begin(), shuffled_indexes.end());

	for(int i = 0; i < 50000; i++) {
		// Calculate gradients and sums it to the overall gradients
		calculate_gradient_based_on_single_training_data(train_inputs[shuffled_indexes[i]], train_actual_outputs[shuffled_indexes[i]]);

		// Change the weights after MINI_BATCH_INTERVAL
		// The following three lines of code is the difference between SGD and GD
		if((i + 1) % (Constants::MINI_BATCH_SIZE) == 0) {
			add_gradient_to_weight_and_biases(Constants::MINI_BATCH_SIZE);
		}
	}
}

void neural_network::add_gradient_to_weight_and_biases(const int MINI_BATCH_SIZE) {
	// average out the gradients and add them to the weights and biases 
	for(int i = 1; i < m_neuron_layers.size() ; i++) {
		for(neuron* ele: m_neuron_layers[i]) {
			ele->average_gradients_and_change_weights_and_bias(MINI_BATCH_SIZE);
		}
	}
}

void neural_network::calculate_gradient_based_on_single_training_data(const std::vector<long double>& a_train_input, int a_actual_output) {
	propagate_forward(a_train_input);
	
	// calculate del_c_by_del_a for output layer
	for(int i = 0; i < m_neuron_layers.back().size(); i++) {
		long double y = (i == a_actual_output) ? 1 : 0;
		long double del_c_by_del_a = 2 * (m_neuron_layers.back()[i]->get_activation() - y);
		m_neuron_layers.back()[i]->set_del_c_by_del_a(del_c_by_del_a);
	}
	// calculate gradients for output layer
	for(int i = 0; i < m_neuron_layers.back().size(); i++) {
		neuron* cur_neuron = m_neuron_layers.back()[i];
		cur_neuron->compute_weight_gradient_based_on_del_c_by_del_a();
		cur_neuron->compute_bias_gradient_based_on_del_c_by_del_a();
	}

	// calculate del_c_by_del_a and gradients for hidden layers
	// I.E do that recursively to propagate backwards
	for(int l = m_neuron_layers.size() - 2; l >= 1; l--) {
		// first calculate del_c_by_del_a
		for(int i = 0; i < m_neuron_layers[l].size(); i++) {
			m_neuron_layers[l][i]->compute_del_c_by_del_a_for_hidden_layer();
		}
		// Then calculate the gradients
		for(int i = 0; i < m_neuron_layers[l].size(); i++) {
			neuron* cur_neuron = m_neuron_layers[l][i];
			cur_neuron->compute_weight_gradient_based_on_del_c_by_del_a();
			cur_neuron->compute_bias_gradient_based_on_del_c_by_del_a();
		}
	}

}

void neural_network::perform_test(const std::vector<std::vector<long double>> &test_inputs, const std::vector<int> &test_actual_outputs) const {
	int number_of_correct = 0;
	for(int i = 0; i < test_inputs.size(); i++) {
		propagate_forward(test_inputs[i]);
		if(get_neural_network_output() == test_actual_outputs[i]) {
			number_of_correct+= 1;
		}
	}
	std::cout << number_of_correct << " / " <<test_inputs.size() << " correct" << std::endl;
}

int neural_network::get_neural_network_output() const {
	int index;
	long double max_value = -100;
	for(int i = 0; i < m_neuron_layers.back().size(); i++) {
		if(m_neuron_layers.back()[i]->get_activation() > max_value) {
			max_value = m_neuron_layers.back()[i]->get_activation();
			index = i;
		}
	}
	return index;
}


int neural_network::give_prediction(const std::vector<long double> &a_input) {
	propagate_forward(a_input);
	return get_neural_network_output();
}