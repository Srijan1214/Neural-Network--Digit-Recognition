#include "neural_network.h"

void neural_network::calculate_first_layer_activation_gradients(){
	for(size_t i = 0; i<m_first_layer_activation_gradients.size(); i++){
		double temp=0;
		m_first_layer_activation_gradients[i]=0;
		//m_first_layer_activation_gradients[i]=sigmoid(m_second_layer_neurons[i]->get_z())*(1-sigmoid(m_second_layer_neurons[i]->get_z()));
		for(size_t j = 0; j<m_second_layer_weights.size(); j++){
			temp=m_first_layer_activation_gradients[i]*m_second_layer_weights[j][i]*m_second_layer_activation_gradients[j]*
				sigmoid(m_second_layer_neurons[j]->get_z())*(1-sigmoid(m_second_layer_neurons[j]->get_z()));
		}
		m_first_layer_activation_gradients[i]+=temp;
	}
}

void neural_network::calculate_second_layer_activation_gradients(){
	for(size_t i=0;i<m_second_layer_activation_gradients.size();i++){
		m_second_layer_activation_gradients[i]=0;
		double temp;
		for(size_t j=0;j<m_output_layer_acitvations.size();j++){
			if(cur_true_result!=j){
				temp=m_output_layer_weights[j][i]*((sigmoid(m_output_layer_neurons[j]->get_z())*(1-sigmoid(m_output_layer_neurons[j]->get_z())))*
					(2*(m_output_layer_acitvations[j]-0.00)));
			} else{
				temp=m_output_layer_weights[j][i]*((sigmoid(m_output_layer_neurons[j]->get_z())*(1-sigmoid(m_output_layer_neurons[j]->get_z())))*
					(2*(m_output_layer_acitvations[j]-1.00)));
			}
			m_second_layer_activation_gradients[i]+=temp;
		}
	}
}

void neural_network::calculate_output_layer_err(){
	int y;
	for(size_t i = 0; i<m_output_layer_err.size(); i++){
		if(i==cur_true_result){
			y=1;
		} else{
			y=0;
		}
		m_output_layer_err[i]=(m_output_layer_acitvations[i]-y)*(sigmoid_prime(m_output_layer_neurons[i]->get_z()));
	}
}

void neural_network::calculate_second_layer_err(){
	for(size_t i=0;i<m_second_layer_err.size();i++){
		double sum=0;
		for(size_t j = 0; j<m_output_layer_err.size(); j++){
			sum+=m_output_layer_weights[j][i]*m_output_layer_err[j];
		}

		m_second_layer_err[i]=(sum)*sigmoid_prime(m_second_layer_neurons[i]->get_z());
	}
}

void neural_network::calculate_first_layer_err(){
	for(size_t i = 0; i<m_first_layer_err.size(); i++){
		double sum=0;
		for(size_t j = 0; j<m_second_layer_weights.size(); j++){
			sum+=m_second_layer_weights[j][i]*m_second_layer_err[j];
		}

		m_first_layer_err[i]=(sum)*sigmoid_prime(m_first_layer_neurons[i]->get_z());
	}
}

void neural_network::calculate_output_layer_weights_gradients(){
	double temp;
	for(size_t i = 0; i<m_output_layer_weights_gradients.size(); i++){
		for(size_t j = 0; j<m_second_layer_weights_gradients.size(); j++){
			////m_output_layer_weights_gradients[i][j]=0;
			////m_output_layer_weights_gradients[i][j]=sigmoid(m_output_layer_neurons[i]->get_z())*(1-sigmoid(m_output_layer_neurons[i]->get_z()))*
			//	//(m_second_layer_acitvations[j]);
			//temp=sigmoid(m_output_layer_neurons[i]->get_z())*(1-sigmoid(m_output_layer_neurons[i]->get_z()))*
			//	(m_second_layer_acitvations[j]);
			//if(i!=cur_true_result){
			//	//m_output_layer_weights_gradients[i][j]*=((2*((m_output_layer_acitvations[i])-0.00)));
			//	temp*=((2*((m_output_layer_acitvations[i])-0.00)));
			//} else{
			//	//m_output_layer_weights_gradients[i][j]*=(2*((m_output_layer_acitvations[i])-1.00));
			//	temp*=((2*((m_output_layer_acitvations[i])-1.00)));
			//}
			////std::cout<<m_output_layer_weights[i][j]<<std::endl;
			////std::cout<<m_output_layer_weights_gradients[i][j]<<std::endl;

			temp=m_second_layer_acitvations[j]*m_output_layer_err[i];
			m_output_layer_weights_gradients[i][j]+=temp;
		}
	}
}

void neural_network::calculate_second_layer_weights_gradients(){
	double temp;
	for(size_t i = 0; i<m_second_layer_weights_gradients.size(); i++){
		for(size_t j = 0; j<m_second_layer_weights_gradients[0].size(); j++){
			//temp=sigmoid(m_second_layer_neurons[i]->get_z())*(1-sigmoid(m_second_layer_neurons[i]->get_z()))*
			//	(m_input_layer_acitvations[j])*m_second_layer_activation_gradients[i];

			temp=m_first_layer_acitvations[j]*m_second_layer_err[i];
			m_second_layer_weights_gradients[i][j]+=temp;
		}
	}
}

void neural_network::calculate_first_layer_weights_gradients(){
	double temp;
	for(size_t i = 0; i<m_first_layer_weights_gradients.size(); i++){
		for(size_t j = 0; j<m_first_layer_weights_gradients[0].size(); j++){
			//m_first_layer_weights_gradients[i][j]=sigmoid(m_first_layer_neurons[i]->get_z())*(1-sigmoid(m_first_layer_neurons[i]->get_z()))*
				//(m_input_layer_acitvations[j])*m_first_layer_activation_gradients[i];
			temp=sigmoid(m_first_layer_neurons[i]->get_z())*(1-sigmoid(m_first_layer_neurons[i]->get_z()))*
				(m_input_layer_acitvations[j])*m_first_layer_activation_gradients[i];
			m_first_layer_weights_gradients[i][j]+=temp;
		}
	}
}

void neural_network::calculate_output_layer_bias_gradients(){
	double temp;
	for(size_t i = 0; i<m_output_layer_bias_gradients.size(); i++){
		////m_output_layer_bias_gradients[i]=sigmoid(m_output_layer_neurons[i]->get_z())*(1-sigmoid(m_output_layer_neurons[i]->get_z()));
		//temp=sigmoid(m_output_layer_neurons[i]->get_z())*(1-sigmoid(m_output_layer_neurons[i]->get_z()));
		//if(i!=cur_true_result){
		//	//m_output_layer_bias_gradients[i]*=(2*((m_output_layer_acitvations[i])-0.00));
		//	temp*=(2*((m_output_layer_acitvations[i])-0.00));
		//} else{
		//	//m_output_layer_bias_gradients[i]*=(2*((m_output_layer_acitvations[i])-1.00));
		//	temp*=(2*((m_output_layer_acitvations[i])-1.00));
		//}

		temp=m_output_layer_err[i];
		m_output_layer_bias_gradients[i]+=temp;
	}
}

void neural_network::calculate_second_layer_bias_gradients(){
	for(size_t i = 0; i<m_second_layer_bias_gradients.size(); i++){
		//m_second_layer_bias_gradients[i]+=sigmoid(m_second_layer_neurons[i]->get_z())*(1-sigmoid(m_second_layer_neurons[i]->get_z()))*
		//	m_second_layer_activation_gradients[i];
		m_second_layer_bias_gradients[i]+=m_second_layer_err[i];
	}
}

void neural_network::calculate_first_layer_bias_gradients(){
	for(size_t i = 0; i<m_first_layer_bias_gradients.size(); i++){
		m_first_layer_bias_gradients[i]+=sigmoid(m_first_layer_neurons[i]->get_z())*(1-sigmoid(m_first_layer_neurons[i]->get_z()))*
			m_first_layer_activation_gradients[i];
	}
}

void neural_network::change_first_layer_weights(){
	for(size_t i = 0; i<m_first_layer_weights.size(); i++){
		for(size_t j= 0; j<m_first_layer_weights[0].size(); j++){
			m_first_layer_weights[i][j]-=m_first_layer_weights_gradients[i][j]*learning_rate;
			m_first_layer_weights_gradients[i][j]=0;
		}
	}
}

void neural_network::change_second_layer_weights(){
	for(size_t i = 0; i<m_second_layer_weights.size(); i++){
		for(size_t j= 0; j<m_second_layer_weights[0].size(); j++){
			m_second_layer_weights[i][j]-=m_second_layer_weights_gradients[i][j]*learning_rate;
			m_second_layer_weights_gradients[i][j]=0;
		}
	}
}

void neural_network::change_output_layer_weights(){
	for(size_t i = 0; i<m_output_layer_weights.size(); i++){
		for(size_t j= 0; j<m_output_layer_weights[0].size(); j++){
			m_output_layer_weights[i][j]-=(m_output_layer_weights_gradients[i][j]*learning_rate);
			m_output_layer_weights_gradients[i][j]=0;
		}
	}
}

void neural_network::change_first_layer_bias(){
	for(size_t i = 0; i<m_first_layer_neurons.size(); i++){
		m_first_layer_neurons[i]->change_bias(m_first_layer_bias_gradients[i]*learning_rate);
		m_first_layer_bias_gradients[i]=0;
	}
}

void neural_network::change_second_layer_bias(){
	for(size_t i = 0; i<m_second_layer_neurons.size(); i++){
		m_second_layer_neurons[i]->change_bias(m_second_layer_bias_gradients[i]*learning_rate);
		m_second_layer_bias_gradients[i]=0;
	}
}

void neural_network::change_output_layer_bias(){
	for(size_t i = 0; i<m_output_layer_neurons.size(); i++){
		m_output_layer_neurons[i]->change_bias(m_output_layer_bias_gradients[i]*learning_rate);
		m_output_layer_bias_gradients[i]=0;
	}
}

void neural_network::average_out_gradients(){
	for(size_t i = 0; i<m_output_layer_weights_gradients.size(); i++){
		for(size_t j = 0; j<m_second_layer_weights_gradients.size(); j++){
			m_output_layer_weights_gradients[i][j]/=m_number_for_SDC;
		}
	}

	for(size_t i = 0; i<m_second_layer_weights_gradients.size(); i++){
		for(size_t j = 0; j<m_second_layer_weights_gradients[0].size(); j++){
			m_second_layer_weights_gradients[i][j]/=m_number_for_SDC;
		}
	}
	for(size_t i = 0; i<m_first_layer_weights_gradients.size(); i++){
		for(size_t j = 0; j<m_first_layer_weights_gradients[0].size(); j++){
			m_first_layer_weights_gradients[i][j]/=m_number_for_SDC;
		}
	}

	for(size_t i = 0; i<m_output_layer_bias_gradients.size(); i++){
		m_output_layer_bias_gradients[i]/=m_number_for_SDC;
	}

	for(size_t i = 0; i<m_second_layer_bias_gradients.size(); i++){
		m_second_layer_bias_gradients[i]/=m_number_for_SDC;
	}

	for(size_t i = 0; i<m_first_layer_bias_gradients.size(); i++){
		m_first_layer_bias_gradients[i]/=m_number_for_SDC;
	}

	m_cost/=m_number_for_SDC;
}

double neural_network::sigmoid(double input){
	return 1/(1+exp(-input));
}

double neural_network::sigmoid_prime(double input){
	return (sigmoid(input)*(1-sigmoid(input)));
}


void neural_network::take_input_and_start_session(std::vector<double>& inputs){
	m_counter++;
	m_input_layer_acitvations=inputs;

	/*for(size_t i=0;i<m_first_layer_neurons.size();i++){
		m_first_layer_neurons[i]->reinitialize_activation(m_first_layer_weights[i],inputs);
		m_first_layer_acitvations[i]=m_first_layer_neurons[i]->get_activation_value();
	}*/

	for(size_t i=0;i<m_second_layer_neurons.size();i++){
		m_second_layer_neurons[i]->reinitialize_activation(m_second_layer_weights[i],inputs);
		m_second_layer_acitvations[i]=m_second_layer_neurons[i]->get_activation_value();
	}

	double max=-1;
	int result;
	for(size_t i=0;i<m_output_layer_neurons.size();i++){
		m_output_layer_neurons[i]->reinitialize_activation(m_output_layer_weights[i],m_second_layer_acitvations);
		m_output_layer_acitvations[i]=m_output_layer_neurons[i]->get_activation_value();
		if(m_output_layer_acitvations[i]>max){
			max=m_output_layer_acitvations[i];
			result=i;
		}
	}

	if(result==cur_true_result){
		correct_values_per_epoc.back()++;
	}
	

	prapagate_backwards();

}

void neural_network::initialize_random_weights(){
	double lower_bound = -1;
	double upper_bound = 1;
	//std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::normal_distribution<double> unif(0.0,1.0);

	std::default_random_engine re;
	re.seed(time(0));

	for(size_t i = 0; i<m_output_layer_weights.size(); i++){
		for(size_t j= 0; j<m_output_layer_weights[0].size(); j++){
			m_output_layer_weights[i][j]=unif(re);
		}
	}

	for(size_t i = 0; i<m_second_layer_weights.size(); i++){
		for(size_t j= 0; j<m_second_layer_weights[0].size(); j++){
			m_second_layer_weights[i][j]=unif(re);
		}
	}

	for(size_t i = 0; i<m_first_layer_weights.size(); i++){
		for(size_t j= 0; j<m_first_layer_weights[0].size(); j++){
			m_first_layer_weights[i][j]=unif(re);
		}
	}

}

void neural_network::initialize_random_activations(){
	double lower_bound = 0;
	double upper_bound = 1;
	//std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::default_random_engine re;
	std::normal_distribution<double> unif(0.0,1.0);
	re.seed(time(0));

	for(size_t i = 0; i<m_output_layer_acitvations.size(); i++){
		m_output_layer_acitvations[i]=unif(re);
	}

	for(size_t i = 0; i<m_second_layer_acitvations.size(); i++){
		m_second_layer_acitvations[i]=unif(re);
	}

	for(size_t i = 0; i<m_first_layer_acitvations.size(); i++){
		m_first_layer_acitvations[i]=unif(re);
	}
}

void neural_network::initialize_random_bias(){
	double lower_bound = -10;
	double upper_bound = 0;
	//std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::normal_distribution<double> unif(0.0,1.0);

	std::default_random_engine re;
	re.seed(time(0));

	for(size_t i=0;i<m_first_layer_neurons.size();i++){
		m_first_layer_neurons[i]->set_bias(unif(re));
	}

	for(size_t i=0;i<m_second_layer_neurons.size();i++){
		m_second_layer_neurons[i]->set_bias(unif(re));
	}

	for(size_t i=0;i<m_output_layer_neurons.size();i++){
		m_output_layer_neurons[i]->set_bias(unif(re));
	}
}

void neural_network::set_layer_size(size_t size){
	m_first_layer_neurons.resize(size,0);
	m_second_layer_neurons.resize(30,0);
	m_output_layer_neurons.resize(10,0);

	m_first_layer_err.resize(size,0);
	m_second_layer_err.resize(30,0);
	m_output_layer_err.resize(10,0);

	m_input_layer_acitvations.resize(size,0);
	m_first_layer_acitvations.resize(size,0);
	m_second_layer_acitvations.resize(30,0);
	m_output_layer_acitvations.resize(10,0);

	m_first_layer_weights.resize(size);
	m_second_layer_weights.resize(30);
	m_output_layer_weights.resize(10);

	for(size_t i=0;i<size;i++){
		m_first_layer_weights[i].resize(size);
	}

	for(size_t i=0;i<30;i++){
		m_second_layer_weights[i].resize(size);
	}

	for(size_t i=0;i<10;i++){
		m_output_layer_weights[i].resize(30);
	}

	m_first_layer_weights_gradients.resize(size);
	m_second_layer_weights_gradients.resize(30);
	m_output_layer_weights_gradients.resize(10);

	for(size_t i=0;i<size;i++){
		m_first_layer_weights_gradients[i].resize(size,0);
	}

	for(size_t i=0;i<30;i++){
		m_second_layer_weights_gradients[i].resize(size,0);
	}

	for(size_t i=0;i<10;i++){
		m_output_layer_weights_gradients[i].resize(30,0);
	}

	m_first_layer_bias_gradients.resize(size,0);
	m_second_layer_bias_gradients.resize(30,0);
	m_output_layer_bias_gradients.resize(10,0);

	m_first_layer_activation_gradients.resize(size);
	m_second_layer_activation_gradients.resize(30);

}

void neural_network::prapagate_backwards(){
	double cost=0;
	for(size_t i=0;i<m_output_layer_acitvations.size();i++){
		if(i!=cur_true_result){
			cost+=(m_output_layer_acitvations[i]-0.00)*(m_output_layer_acitvations[i]-0.00);
		} else{
			cost+=(m_output_layer_acitvations[i]-1.00)*(m_output_layer_acitvations[i]-1.00);
		}
	}
	m_cost+=(cost/2);

	calculate_output_layer_err();
	calculate_second_layer_err();

	calculate_output_layer_weights_gradients();
	calculate_output_layer_bias_gradients();
	if(m_counter%m_number_for_SDC==0){
		average_out_gradients();
		change_output_layer_weights();
		change_output_layer_bias();
	}

	calculate_second_layer_activation_gradients();
	calculate_second_layer_weights_gradients();
	calculate_second_layer_bias_gradients();
	if(m_counter%m_number_for_SDC==0){
		change_second_layer_weights();
		change_second_layer_bias();
	}

	/*calculate_first_layer_activation_gradients();
	calculate_first_layer_weights_gradients();
	calculate_first_layer_bias_gradients();
	if(m_counter%m_number_for_SDC==0){
		change_first_layer_weights();
		change_first_layer_bias();
	}*/


}

void neural_network::show_output_layer(){
	for(size_t i=0;i<m_output_layer_acitvations.size();i++){
		std::cout<<i<<" "<<m_output_layer_acitvations[i]<<std::endl;
	}
}

void neural_network::create_neuron_objects(){
	for(size_t i=0;i<m_first_layer_neurons.size();i++){
		m_first_layer_neurons[i]=new neuron(m_first_layer_weights[i],m_first_layer_acitvations);
	}

	for(size_t i=0;i<m_second_layer_neurons.size();i++){
		m_second_layer_neurons[i]=new neuron(m_second_layer_weights[i],m_input_layer_acitvations);
	}

	for(size_t i=0;i<m_output_layer_neurons.size();i++){
		m_output_layer_neurons[i]=new neuron(m_output_layer_weights[i],m_second_layer_acitvations);
	}
}

void neural_network::create_necessary_stuff(){
	m_counter=0;
	learning_rate=3;

	initialize_random_weights();
	//initialize_random_activations();
	create_neuron_objects();
	initialize_random_bias();
}

void neural_network::set_cur_true_result(int input){
	cur_true_result=input;
}

void neural_network::show_cost(){
	double cost=0;
	for(size_t i=0;i<m_output_layer_acitvations.size();i++){
		if(i!=cur_true_result){
			cost+=(m_output_layer_acitvations[i]-0.00)*(m_output_layer_acitvations[i]-0.00);
		} else{
			cost+=(m_output_layer_acitvations[i]-1.00)*(m_output_layer_acitvations[i]-1.00);
		}
	}
	//std::cout<<"cost="<<cost<<std::endl;
	std::cout<<"cost="<<m_cost<<std::endl;
	m_cost=0;

	show_input_and_true_result();

}

void neural_network::show_input_and_true_result(){
	for(size_t r = 0; r<28; r++){
		for(size_t c = 0; c<28; c++){
			std::cout<<m_input_layer_acitvations[r*28+c];
		}
		std::cout<<std::endl;
	}
	std::cout<<cur_true_result<<std::endl<<std::endl<<std::endl;
}

void neural_network::set_number_for_SDC(int input){
	m_number_for_SDC=input;
}

void neural_network::add_epoc(){
	epoc++;
	correct_values_per_epoc.push_back(0);
}

void neural_network::print_test_results(){
	/*for(size_t i = 0; i<correct_values_per_epoc.size(); i++){
		std::cout<<"Epoch: "<<i<<"---"<<correct_values_per_epoc[i]<<"/10000"<<std::endl;
	}
*/
	std::cout<<"Epoch: "<<correct_values_per_epoc.size()<<"---"<<correct_values_per_epoc.back()<<"/10000"<<std::endl;

}

void neural_network::perform_test(std::vector<std::vector<double>>&inputs,std::vector<int> & values){
	int correct_results=0;
	for(size_t i = 0; i<inputs.size(); i++){
		m_input_layer_acitvations=inputs[i];

		for(size_t i=0;i<m_second_layer_neurons.size();i++){
			m_second_layer_neurons[i]->reinitialize_activation(m_second_layer_weights[i],m_input_layer_acitvations);
			m_second_layer_acitvations[i]=m_second_layer_neurons[i]->get_activation_value();
		}

		double max=-1;
		int result;
		for(size_t i=0;i<m_output_layer_neurons.size();i++){
			m_output_layer_neurons[i]->reinitialize_activation(m_output_layer_weights[i],m_second_layer_acitvations);
			m_output_layer_acitvations[i]=m_output_layer_neurons[i]->get_activation_value();
			if(m_output_layer_acitvations[i]>max){
				max=m_output_layer_acitvations[i];
				result=i;
			}
		}

		if(result==values[i]){
			correct_results++;
		}
	}

	std::cout<<"Epoch: "<<epoc<<" Fraction of correct: "<<correct_results<<"/ "<<inputs.size()<<std::endl;

}
