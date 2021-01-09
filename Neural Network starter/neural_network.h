#pragma once
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <assert.h>
#include "neuron.h"
#include "constants.h"
#include "random_engine.h"

class neural_network {
   private:

	std::vector<std::vector<neuron *>> m_neuron_layers;

	void show_input_and_true_result();

	int get_neural_network_output() const;

	void propagate_forward(const std::vector<long double> &a_input) const;
	void add_gradient_to_weight_and_biases(const int MINI_BATCH_SIZE);
	void calculate_gradient_based_on_single_training_data(const std::vector<long double>& a_train_input, int a_actual_output);

   public:
	neural_network(int a_input_dimension, int a_no_of_classifications);

	void perform_SGD_training(std::vector<std::vector<long double>> &train_inputs, std::vector<int> &train_actual_outputs);

	void perform_test(const std::vector<std::vector<long double>> &test_inputs,const std::vector<int> &test_actual_outputs) const;
	int give_prediction(const std::vector<long double> &a_input);
};
