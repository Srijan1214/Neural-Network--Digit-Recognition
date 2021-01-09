#pragma once
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

class mnist_parser {
   public:
	unsigned char* read_mnist_labels(std::string full_path,
									 int& number_of_labels);
	int reverseInt(int i);
	void read_mnist_train();
	void read_mnist_train(std::vector<std::vector<long double>>& train_inputs,
						  std::vector<int>& train_actual_outputs);
	void read_mnist_test(std::vector<std::vector<long double>>& test_inputs,
						 std::vector<int>& test_actual_outputs);
};