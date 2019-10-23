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

void neural_network::calculate_output_layer_weights_gradients(){
	double temp;
	for(size_t i = 0; i<m_output_layer_weights_gradients.size(); i++){
		for(size_t j = 0; j<m_second_layer_weights_gradients.size(); j++){
			//m_output_layer_weights_gradients[i][j]=0;
			//m_output_layer_weights_gradients[i][j]=sigmoid(m_output_layer_neurons[i]->get_z())*(1-sigmoid(m_output_layer_neurons[i]->get_z()))*
				//(m_second_layer_acitvations[j]);
			temp=sigmoid(m_output_layer_neurons[i]->get_z())*(1-sigmoid(m_output_layer_neurons[i]->get_z()))*
				(m_second_layer_acitvations[j]);
			if(i!=cur_true_result){
				//m_output_layer_weights_gradients[i][j]*=((2*((m_output_layer_acitvations[i])-0.00)));
				temp*=((2*((m_output_layer_acitvations[i])-0.00)));
			} else{
				//m_output_layer_weights_gradients[i][j]*=(2*((m_output_layer_acitvations[i])-1.00));
				temp*=((2*((m_output_layer_acitvations[i])-1.00)));
			}
			//std::cout<<m_output_layer_weights[i][j]<<std::endl;
			//std::cout<<m_output_layer_weights_gradients[i][j]<<std::endl;
			m_output_layer_weights_gradients[i][j]+=temp;
		}
	}
}

void neural_network::calculate_second_layer_weights_gradients(){
	double temp;
	for(size_t i = 0; i<m_second_layer_weights_gradients.size(); i++){
		for(size_t j = 0; j<m_second_layer_weights_gradients[0].size(); j++){
			//m_second_layer_weights_gradients[i][j]=sigmoid(m_second_layer_neurons[i]->get_z())*(1-sigmoid(m_second_layer_neurons[i]->get_z()))*
				//(m_first_layer_acitvations[j])*m_second_layer_activation_gradients[i];
			temp=sigmoid(m_second_layer_neurons[i]->get_z())*(1-sigmoid(m_second_layer_neurons[i]->get_z()))*
				(m_first_layer_acitvations[j])*m_second_layer_activation_gradients[i];
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
		//m_output_layer_bias_gradients[i]=sigmoid(m_output_layer_neurons[i]->get_z())*(1-sigmoid(m_output_layer_neurons[i]->get_z()));
		temp=sigmoid(m_output_layer_neurons[i]->get_z())*(1-sigmoid(m_output_layer_neurons[i]->get_z()));
		if(i!=cur_true_result){
			//m_output_layer_bias_gradients[i]*=(2*((m_output_layer_acitvations[i])-0.00));
			temp*=(2*((m_output_layer_acitvations[i])-0.00));
		} else{
			//m_output_layer_bias_gradients[i]*=(2*((m_output_layer_acitvations[i])-1.00));
			temp*=(2*((m_output_layer_acitvations[i])-1.00));
		}
		m_output_layer_bias_gradients[i]+=temp;
	}
}

void neural_network::calculate_second_layer_bias_gradients(){
	for(size_t i = 0; i<m_second_layer_bias_gradients.size(); i++){
		m_second_layer_bias_gradients[i]+=sigmoid(m_second_layer_neurons[i]->get_z())*(1-sigmoid(m_second_layer_neurons[i]->get_z()))*
			m_second_layer_activation_gradients[i];
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
			m_first_layer_weights[i][j]-=m_first_layer_weights_gradients[i][j];
			m_first_layer_weights_gradients[i][j]=0;
		}
	}
}

void neural_network::change_second_layer_weights(){
	for(size_t i = 0; i<m_second_layer_weights.size(); i++){
		for(size_t j= 0; j<m_second_layer_weights[0].size(); j++){
			m_second_layer_weights[i][j]-=m_second_layer_weights_gradients[i][j];
			m_second_layer_weights_gradients[i][j]=0;
		}
	}
}

void neural_network::change_output_layer_weights(){
	for(size_t i = 0; i<m_output_layer_weights.size(); i++){
		for(size_t j= 0; j<m_output_layer_weights[0].size(); j++){
			m_output_layer_weights[i][j]-=m_output_layer_weights_gradients[i][j];
			m_output_layer_weights_gradients[i][j]=0;
		}
	}
}

void neural_network::change_first_layer_bias(){
	for(size_t i = 0; i<m_first_layer_neurons.size(); i++){
		m_first_layer_neurons[i]->change_bias(m_first_layer_bias_gradients[i]);
		m_first_layer_bias_gradients[i]=0;
	}
}

void neural_network::change_second_layer_bias(){
	for(size_t i = 0; i<m_second_layer_neurons.size(); i++){
		m_second_layer_neurons[i]->change_bias(m_second_layer_bias_gradients[i]);
		m_second_layer_bias_gradients[i]=0;
	}
}

void neural_network::change_output_layer_bias(){
	for(size_t i = 0; i<m_output_layer_neurons.size(); i++){
		m_output_layer_neurons[i]->change_bias(m_output_layer_bias_gradients[i]);
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

void neural_network::take_input_and_start_session(std::vector<double>& inputs){
	m_counter++;
	m_input_layer_acitvations=inputs;

	for(size_t i=0;i<m_first_layer_neurons.size();i++){
		m_first_layer_neurons[i]->reinitialize_activation(m_first_layer_weights[i],inputs);
		m_first_layer_acitvations[i]=m_first_layer_neurons[i]->get_activation_value();
	}	

	for(size_t i=0;i<m_second_layer_neurons.size();i++){
		m_second_layer_neurons[i]->reinitialize_activation(m_second_layer_weights[i],m_first_layer_acitvations);
		m_second_layer_acitvations[i]=m_second_layer_neurons[i]->get_activation_value();
	}

	for(size_t i=0;i<m_output_layer_neurons.size();i++){
		m_output_layer_neurons[i]->reinitialize_activation(m_output_layer_weights[i],m_second_layer_acitvations);
		m_output_layer_acitvations[i]=m_output_layer_neurons[i]->get_activation_value();
	}

	prapagate_backwards();
	
}

void neural_network::initialize_random_weights(){
	double lower_bound = -3.5;
	double upper_bound = 3;
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
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
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
	std::default_random_engine re;
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
	std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
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
	m_second_layer_neurons.resize(size,0);
	m_output_layer_neurons.resize(10,0);

	m_input_layer_acitvations.resize(size,0);
	m_first_layer_acitvations.resize(size,0);
	m_second_layer_acitvations.resize(size,0);
	m_output_layer_acitvations.resize(10,0);

	m_first_layer_weights.resize(size);
	m_second_layer_weights.resize(size);
	m_output_layer_weights.resize(10);

	for(size_t i=0;i<size;i++){
		m_first_layer_weights[i].resize(size);
	}

	for(size_t i=0;i<size;i++){
		m_second_layer_weights[i].resize(size);
	}

	for(size_t i=0;i<10;i++){
		m_output_layer_weights[i].resize(size);
	}

	m_first_layer_weights_gradients.resize(size);
	m_second_layer_weights_gradients.resize(size);
	m_output_layer_weights_gradients.resize(10);
	
	for(size_t i=0;i<size;i++){
		m_first_layer_weights_gradients[i].resize(size,0);
	}

	for(size_t i=0;i<size;i++){
		m_second_layer_weights_gradients[i].resize(size,0);
	}

	for(size_t i=0;i<10;i++){
		m_output_layer_weights_gradients[i].resize(size,0);
	}

	m_first_layer_bias_gradients.resize(size,0);
	m_second_layer_bias_gradients.resize(size,0);
	m_output_layer_bias_gradients.resize(10,0);

	m_first_layer_activation_gradients.resize(size);
	m_second_layer_activation_gradients.resize(size);

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
	m_cost+=cost;

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
	calculate_first_layer_activation_gradients();
	calculate_first_layer_weights_gradients();
	calculate_first_layer_bias_gradients();
	if(m_counter%m_number_for_SDC==0){
		change_first_layer_weights();
		change_first_layer_bias();
	}


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
		m_second_layer_neurons[i]=new neuron(m_second_layer_weights[i],m_second_layer_acitvations);
	}

	for(size_t i=0;i<m_output_layer_neurons.size();i++){
		m_output_layer_neurons[i]=new neuron(m_output_layer_weights[i],m_second_layer_acitvations);
	}
}

void neural_network::create_necessary_stuff(){
	m_counter=0;
	initialize_random_weights();
	initialize_random_activations();
	create_neuron_objects();
	//initialize_random_bias();
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
}

void neural_network::set_number_for_SDC(int input){
	m_number_for_SDC=input;
}
