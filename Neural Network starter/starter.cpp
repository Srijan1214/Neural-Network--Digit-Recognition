#include <fstream>
#include <iostream>
#include "neural_network.h"
#include <vector>
#include <algorithm>
#include <random>

using namespace std;

neural_network neural_network_obj;

unsigned char* read_mnist_labels(string full_path,int& number_of_labels){
	auto reverseInt = [](int i){
		unsigned char c1,c2,c3,c4;
		c1 = i&255,c2 = (i>>8)&255,c3 = (i>>16)&255,c4 = (i>>24)&255;
		return ((int)c1<<24)+((int)c2<<16)+((int)c3<<8)+c4;
	};

	typedef unsigned char uchar;

	ifstream file(full_path,ios::binary);

	if(file.is_open()){
		int magic_number = 0;
		file.read((char *)&magic_number,sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if(magic_number!=2049) throw runtime_error("Invalid MNIST label file!");

		file.read((char *)&number_of_labels,sizeof(number_of_labels)),number_of_labels = reverseInt(number_of_labels);
		uchar* _dataset = new uchar[number_of_labels];
		for(int i = 0; i<number_of_labels; i++){
			file.read((char*)&_dataset[i],1);
		}
		return _dataset;
	} else{
		throw runtime_error("Unable to open file `"+full_path+"`!");
	}
}
unsigned char* asd;
vector<int> value_list;
vector<vector<double>> grey_scale_values;
vector<int> test_value_list;
vector<vector<double>> test_grey_scale_values;
auto rng = std::default_random_engine{};

int reverseInt(int i){
	unsigned char c1,c2,c3,c4;

	c1 = i&255;
	c2 = (i>>8)&255;
	c3 = (i>>16)&255;
	c4 = (i>>24)&255;

	return ((int)c1<<24)+((int)c2<<16)+((int)c3<<8)+c4;
}
void read_mnist(){
	int x=10000;
	asd=read_mnist_labels("train-labels.idx1-ubyte",x);
	for(int i=0;i<60000;i++){
		value_list.push_back((int)(asd[i]));
		if(value_list.back()>9){
			cerr<<value_list.back();
			exit(1);
		}
	}

	asd=read_mnist_labels("t10k-labels.idx1-ubyte",x);
	for(int i=0;i<10000;i++){
		test_value_list.push_back((int)(asd[i]));
		if(test_value_list.back()>9){
			cerr<<test_value_list.back();
			exit(1);
		}
	}
	delete[] asd;

	ifstream file("train-images.idx3-ubyte",ios::binary);
	if(file.is_open()){
		int magic_number=0;
		int number_of_images=0;
		int n_rows=0;
		int n_cols=0;
		file.read((char*)&magic_number,sizeof(magic_number));
		magic_number= reverseInt(magic_number);
		file.read((char*)&number_of_images,sizeof(number_of_images));
		number_of_images= reverseInt(number_of_images);
		file.read((char*)&n_rows,sizeof(n_rows));
		n_rows= reverseInt(n_rows);
		file.read((char*)&n_cols,sizeof(n_cols));
		n_cols= reverseInt(n_cols);	
		vector<double> inputs;
		for(int i=0;i<number_of_images;++i){
			for(int r=0;r<n_rows;++r){
				for(int c=0;c<n_cols;++c){
					unsigned char temp=0;
					file.read((char*)&temp,sizeof(temp));
					inputs.push_back((int)temp);
				}
			}
			grey_scale_values.push_back(inputs);
			inputs.clear();
		}
	} else{
		cout<<"not open";
	}

	{
		ifstream file("t10k-images.idx3-ubyte",ios::binary);
		if(file.is_open()){
			int magic_number=0;
			int number_of_images=0;
			int n_rows=0;
			int n_cols=0;
			file.read((char*)&magic_number,sizeof(magic_number));
			magic_number= reverseInt(magic_number);
			file.read((char*)&number_of_images,sizeof(number_of_images));
			number_of_images= reverseInt(number_of_images);
			file.read((char*)&n_rows,sizeof(n_rows));
			n_rows= reverseInt(n_rows);
			file.read((char*)&n_cols,sizeof(n_cols));
			n_cols= reverseInt(n_cols);
			vector<double> inputs;
			for(int i=0;i<number_of_images;++i){
				for(int r=0;r<n_rows;++r){
					for(int c=0;c<n_cols;++c){
						unsigned char temp=0;
						file.read((char*)&temp,sizeof(temp));
						inputs.push_back((int)temp);
					}
				}
				test_grey_scale_values.push_back(inputs);
				inputs.clear();
			}
		} else{
			cout<<"not open";
		}
	}
}

void perform_neural_network_computation(){
	const int SDC_INTERVAL=10;	//Interation after to compute Stocastic Gradinet Descent

	neural_network_obj.set_number_for_SDC(SDC_INTERVAL);

	//shuffle
	std::vector<int> indexes;
	indexes.reserve(value_list.size());
	for(int i = 0; i<value_list.size(); ++i){
		indexes.push_back(i);
	}

	std::random_shuffle(indexes.begin(),indexes.end());
	neural_network_obj.add_epoc();
	for(int m_counter = 0; m_counter<value_list.size(); m_counter++){
		int value=value_list[indexes[m_counter]];
		neural_network_obj.set_cur_true_result(value);
		neural_network_obj.take_input_and_start_session(grey_scale_values[indexes[m_counter]]);
		if((m_counter+1)%SDC_INTERVAL==0){
			//neural_network_obj.show_output_layer();
			//neural_network_obj.show_cost();
		}
	}
}

int main(){
	int x=10000;

	read_mnist();
	neural_network_obj.set_layer_size(28*28);
	neural_network_obj.create_necessary_stuff();

	////Testing if the image matches the values
	//for(size_t i = 0; i<value_list.size(); i++){
	//	cout<<value_list[i]<<endl<<endl<<endl;

	//	for(size_t r = 0; r<28; r++){
	//		for(size_t c = 0; c<28; c++){
	//			cout<<grey_scale_values[i][r*28+c];
	//		}
	//		cout<<endl;
	//	}
	//	cout<<endl<<endl<<endl;
	//}

	for(int i=0;i<1000;i++){
		perform_neural_network_computation();
		//cout<<"Epoch: "<<i<<endl;
		//neural_network_obj.print_test_results();
		//cin>>x;
		neural_network_obj.perform_test(test_grey_scale_values,test_value_list);
	}
	cin>>x;
}