#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "mnist_parser.h"
#include "neural_network.h"

using namespace std;

void write_predictions_to_file(int a_epoch, neural_network& a_neural_net_obj,std::vector<std::vector<long double>>& train_inputs, std::vector<int>& train_actual_outputs) {
	string file_name = "Epoch" + to_string(a_epoch) + " Testing";
	std::cout << "Writing Predictions to the file: " << file_name << std::endl << std::endl;
	ofstream fstream_obj(file_name);
	if(!fstream_obj.is_open()){
		cerr << "Unable to open file" << endl;
		return;
	}

	// The following code prints out the training inputs in terms of dots so that it looks like the number
	for (size_t i = 0; i < train_actual_outputs.size(); i++) {
		fstream_obj << "The number:           " << train_actual_outputs[i] << "\n";
		fstream_obj << "NeuralNet prediction: " << a_neural_net_obj.give_prediction(train_inputs[i]) << "\n" << "\n";

		for (size_t r = 0; r < 28; r++) {
			for (size_t c = 0; c < 28; c++) {
				if (train_inputs[i][r * 28 + c] == 0) {
					fstream_obj << " ";
				} else {
					fstream_obj << ".";
				}
			}
			fstream_obj << "\n";
		}
		fstream_obj << "\n" << "---------------------------------------------------------" << "\n";
	}
	fstream_obj.close();
}

int main() {	
	vector<vector<long double>> test_inputs;
	vector<int> test_actual_outputs;
	vector<vector<long double>> train_inputs;
	vector<int> train_actual_outputs;

	// Parsing the test train input gotten from the MNIST Website
	// and reading them inside vectors
	mnist_parser mnist_parser_obj;
	mnist_parser_obj.read_mnist_test(test_inputs, test_actual_outputs);
	mnist_parser_obj.read_mnist_train(train_inputs, train_actual_outputs);

	// Create a neural_network_obj and set the input and output layer sizes
	neural_network neural_network_obj(train_inputs[0].size(), 10);

	// The loop to train the Neural Network and test it along the way
	for (int epoch = 1; epoch < 100000; epoch++) {
		std::cout << "Epoch: " << epoch << " training" << std::endl;
		neural_network_obj.perform_SGD_training(train_inputs, train_actual_outputs);
		std::cout << "Epoch: " << epoch << " testing       ";
		neural_network_obj.perform_test(test_inputs, test_actual_outputs);
		if(Constants::SHOULD_WRITE_TO_FILE){
			write_predictions_to_file(epoch, neural_network_obj,test_inputs, test_actual_outputs);
		}
	}
}