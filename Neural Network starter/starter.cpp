#include <fstream>
#include <iostream>
#include "neural_network.h"
#include <vector>

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
vector<int> number_list;

int reverseInt(int i){
	unsigned char c1,c2,c3,c4;

	c1 = i&255;
	c2 = (i>>8)&255;
	c3 = (i>>16)&255;
	c4 = (i>>24)&255;

	return ((int)c1<<24)+((int)c2<<16)+((int)c3<<8)+c4;
}
void read_mnist(){
	int m_counter=0;
	cout<<m_counter<<endl;



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
		//cout<<n_rows<<endl;
		//cout<<n_cols<<endl;
		//cout<<number_of_images<<endl;
		vector<double> inputs;
		neural_network_obj.set_number_for_SDC(100);
		for(int i=0;i<number_of_images;++i){
			for(int r=0;r<n_rows;++r){
				for(int c=0;c<n_cols;++c){
					unsigned char temp=0;
					file.read((char*)&temp,sizeof(temp));
					//cout<<(int)temp;
					inputs.push_back((int)temp);
				}
				//cout<<endl;
			}
			//cout<<endl;
			neural_network_obj.set_cur_true_result(number_list[m_counter++]);
			neural_network_obj.take_input_and_start_session(inputs);
			if((i+1)%100==0){
				neural_network_obj.show_output_layer();
				neural_network_obj.show_cost();
				cout<<"-------Number of train inputs in this epoc: "<<i+1<<"--"<<number_list[m_counter]<<endl<<endl;
			}
			inputs.clear();
		}
	} else{
		cout<<"not open";
	}
}

int main(){
	int x=10000;
	//neural_network_obj.initialize_random_weights();
	asd=read_mnist_labels("t10k-labels.idx1-ubyte",x);
	neural_network_obj.set_layer_size(28*28);
	neural_network_obj.create_necessary_stuff();

	for(int i=0;i<10000;i++){
		number_list.push_back((int)(asd[i]));
		if(number_list.back()>9){
			cout<<number_list.back();
			exit(1);
		}
	}

	delete[] asd;

	for(int i=0;i<35;i++)
		read_mnist();
	//for(int i=0;i<10000;i++){
		//cout<<(int)(asd[5102])<<endl;
	//}
	cin>>x;
}