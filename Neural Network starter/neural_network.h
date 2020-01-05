#pragma once
#include <vector>
#include "neuron.h"
#include <random>
#include <chrono> 
#include <iostream>

class neural_network {
private:
	
	std::vector<int> correct_values_per_epoc;

	std::vector<neuron*> m_first_layer_neurons;
	std::vector<neuron*> m_second_layer_neurons;
	std::vector<neuron*> m_output_layer_neurons;

	std::vector<double> m_first_layer_err;
	std::vector<double> m_second_layer_err;
	std::vector<double> m_output_layer_err;

	std::vector<double> m_input_layer_acitvations;
	std::vector<double> m_first_layer_acitvations;
	std::vector<double> m_second_layer_acitvations;
	std::vector<double> m_output_layer_acitvations;

	std::vector<std::vector<double>> m_first_layer_weights;
	std::vector<std::vector<double>> m_second_layer_weights;
	std::vector<std::vector<double>> m_output_layer_weights;

	std::vector<std::vector<double>> m_first_layer_weights_gradients;
	std::vector<std::vector<double>> m_second_layer_weights_gradients;
	std::vector<std::vector<double>> m_output_layer_weights_gradients;

	std::vector<double> m_first_layer_bias_gradients;
	std::vector<double> m_second_layer_bias_gradients;
	std::vector<double> m_output_layer_bias_gradients;

	std::vector<double> m_first_layer_activation_gradients;
	std::vector<double> m_second_layer_activation_gradients;

	double m_cost;
	int cur_true_result;
	int m_number_for_SDC;
	int m_counter;
	int epoc;
	
	double learning_rate;

	void calculate_first_layer_activation_gradients();
	void calculate_second_layer_activation_gradients();

	void calculate_output_layer_weights_gradients();
	void calculate_second_layer_weights_gradients();
	void calculate_first_layer_weights_gradients();

	void calculate_output_layer_bias_gradients();
	void calculate_second_layer_bias_gradients();
	void calculate_first_layer_bias_gradients();

	void calculate_output_layer_err();
	void calculate_second_layer_err();
	void calculate_first_layer_err();


	void change_first_layer_weights();
	void change_second_layer_weights();
	void change_output_layer_weights();

	void change_first_layer_bias();
	void change_second_layer_bias();
	void change_output_layer_bias();

	void average_out_gradients();

	void show_input_and_true_result();

	double sigmoid(double input);
	double sigmoid_prime(double input);

public:
	void initialize_random_weights();
	void initialize_random_activations();
	void initialize_random_bias();
	void set_layer_size(size_t);
	void take_input_and_start_session(std::vector<double> &);
	void prapagate_backwards();
	void show_output_layer();
	void create_neuron_objects();
	void create_necessary_stuff();
	void set_cur_true_result(int);
	void show_cost();
	void set_number_for_SDC(int);
	void add_epoc();
	void print_test_results();
	void perform_test(std::vector<std::vector<double>> &,std::vector<int> &);


};
