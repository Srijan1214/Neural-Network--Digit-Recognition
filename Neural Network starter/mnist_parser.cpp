#include "mnist_parser.h"

using namespace std;

unsigned char* mnist_parser::read_mnist_labels(std::string full_path,
											   int& number_of_labels) {
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255,
		c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	typedef unsigned char uchar;

	ifstream file(full_path, ios::binary);

	if (file.is_open()) {
		int magic_number = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2049)
			throw runtime_error("Invalid MNIST label file!");

		file.read((char*)&number_of_labels, sizeof(number_of_labels)),
			number_of_labels = reverseInt(number_of_labels);
		uchar* _dataset = new uchar[number_of_labels];
		for (int i = 0; i < number_of_labels; i++) {
			file.read((char*)&_dataset[i], 1);
		}
		return _dataset;
	} else {
		throw runtime_error("Unable to open file `" + full_path + "`!");
	}
}

int mnist_parser::reverseInt(int i) {
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void mnist_parser::read_mnist_train(
	std::vector<std::vector<long double>>& train_inputs,
	std::vector<int>& train_actual_outputs) {
	unsigned char* asd;

	int x = 10000;
	asd = read_mnist_labels("train-labels.idx1-ubyte", x);
	for (int i = 0; i < 60000; i++) {
		train_actual_outputs.push_back((int)(asd[i]));
		if (train_actual_outputs.back() > 9) {
			cerr << train_actual_outputs.back();
			exit(1);
		}
	}
	delete[] asd;

	ifstream file("train-images.idx3-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);
		vector<long double> inputs;
		for (int i = 0; i < number_of_images; ++i) {
			for (int r = 0; r < n_rows; ++r) {
				for (int c = 0; c < n_cols; ++c) {
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					inputs.push_back((int)temp);
				}
			}
			train_inputs.push_back(inputs);
			inputs.clear();
		}
	} else {
		cout << "not open";
	}
}

void mnist_parser::read_mnist_test(
	std::vector<std::vector<long double>>& test_inputs,
	std::vector<int>& test_actual_outputs) {
	unsigned char* asd;

	int x = 10000;
	asd = read_mnist_labels("t10k-labels.idx1-ubyte", x);
	for (int i = 0; i < 10000; i++) {
		test_actual_outputs.push_back((int)(asd[i]));
		if (test_actual_outputs.back() > 9) {
			cerr << test_actual_outputs.back();
			exit(1);
		}
	}

	ifstream file("t10k-images.idx3-ubyte", ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = reverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);
		vector<long double> inputs;
		for (int i = 0; i < number_of_images; ++i) {
			for (int r = 0; r < n_rows; ++r) {
				for (int c = 0; c < n_cols; ++c) {
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					inputs.push_back((int)temp);
				}
			}
			test_inputs.push_back(inputs);
			inputs.clear();
		}
	} else {
		cout << "not open";
	}
}
