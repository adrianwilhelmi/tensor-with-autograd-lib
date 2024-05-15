#include<fstream>
#include<string>
#include<iostream>
#include<algorithm>
#include<vector>
#include<stdexcept>

#include"tensor_lib.hpp"

template<typename T = float>
std::pair<Tensor<T>, Tensor<T>> csv_to_tensors(const std::string& file_path,
		std::size_t num_imgs){
	std::ifstream file(file_path);
	if(!file.is_open())
		throw std::runtime_error("failed to open file: " + file_path);

	//std::vector<Tensor<float>> imgs;
	//std::vector<Tensor<float>> labels;

	Tensor<float> imgs(num_imgs, 28 * 28);
	Tensor<float> labels(num_imgs, 1, 10);

	std::string row;
	std::getline(file, row);
 
	std::size_t count = 0;
	while(std::getline(file, row) && count < num_imgs){
		std::istringstream ss(row);
		std::string token;
		Tensor<T> img(28*28);
		img.enable_grad();
		int j = 0;

		while(std::getline(ss, token, ',')){
			if(j == 0){
				Tensor<int> temp = std::stoi(token);
				Tensor<float> t = temp.one_hot<float>(10);

				labels[count] += t;
				//labels.push_back(t);
			}
			else{
				img.data()[(j - 1)] = std::stoi(token) / 255.0;
			}
			++j;
		}

		imgs[count] += img;
		//imgs.push_back(std::move(img));
   
		++count;
	}
	
	return std::make_pair(std::move(imgs), std::move(labels));
}


template<typename T>
Tensor<T> forward(
		const Tensor<T>& X,
		Tensor<T>& W1,
		Tensor<T>& B1,
		Tensor<T>& W2,
		Tensor<T>& B2
		){
	Tensor<T> hmm = tensor::matmul(X, W1);
	Tensor<T> hmmb = hmm + B1;
	Tensor<T> hidden = hmmb.relu();

	Tensor<T> omm = tensor::matmul(hidden, W2);
	return omm + B2;
}

void mnist(){
	//load imgs into tensors

	const std::string train_path = "./examples/dataset/mnist_train.csv";
	const std::string test_path = "./examples/dataset/mnist_test.csv";

	const std::size_t num_imgs = 2000;

	auto [X, Y] = csv_to_tensors<float>(train_path, num_imgs);
	auto [X_test, Y_test] = csv_to_tensors<float>(test_path, 
			10 * num_imgs);



	std::cout << "X shape" << std::endl;
	std::cout << X.descriptor() << std::endl;

	std::cout << "Y shape" << std::endl;
	std::cout << Y.descriptor() << std::endl;

	//initialize net params

	const std::size_t input_size = 28 * 28;
	const std::size_t hidden_size = 32;
	const std::size_t output_classes = 10;

	Tensor<float> W1 = tensor::random_normal<float>(
			0, 0.01, input_size, hidden_size);
	Tensor<float> B1 = tensor::zeros<float>(1, hidden_size);
	Tensor<float> W2 = tensor::random_normal<float>(
			0, 0.01, hidden_size, output_classes);
	Tensor<float> B2 = tensor::zeros<float>(1, output_classes);

	W1.enable_grad();
	B1.enable_grad();
	W2.enable_grad();
	B2.enable_grad();

	const std::size_t num_epochs = 20;
	float learning_rate = 0.1;
	
	const std::size_t batch_size = 50;
	const std::size_t num_batches = num_imgs / batch_size;



	//learning
	
	auto start = std::chrono::high_resolution_clock::now();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	for(std::size_t epoch = 0; epoch < num_epochs; ++epoch){
		std::cout << "epoch: " << epoch <<  std::endl; 


		start = std::chrono::high_resolution_clock::now();
		
		if(epoch == 30)
			learning_rate /= 10.0;


		for(std::size_t batch = 0; batch < num_batches; ++batch){

			//minibatch

			/*
			std::size_t start_index = tensor::randint(0, 
					num_imgs - batch_size - 1, 1).item();
			*/

			std::size_t start_index = batch * batch_size;

			Tensor<float> X_batch = X.dimslices(0, 
					start_index, start_index + batch_size);
			Tensor<float> Y_batch = Y.dimslices(0,
					start_index, start_index + batch_size);


			//Tensor<float> X_flat = X[i].reshape(1,28*28);
			auto output = forward<float>(
					X_batch,
					W1,
					B1,
					W2,
					B2
					);

			auto loss = tensor::cross_entropy<float>(output, 
								Y_batch);
			loss.backward();
			//std::cout << "loss: " << loss;

			W2 -= learning_rate * W2.grad();
			B2 -= learning_rate * B2.grad();
			W1 -= learning_rate * W1.grad();
			B1 -= learning_rate * B1.grad();

			W2.zero_grad();
			B2.zero_grad();
			W1.zero_grad();
			B1.zero_grad();
		}

		stop = std::chrono::high_resolution_clock::now();

		duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
		std::cout << "duration: " << duration.count() << std::endl;
	}





	//testing
	
	std::size_t success = 0;

	for(std::size_t i = 0; i < Y_test.size(); ++i){
		//Tensor<float> X_flat = test_img[i].reshape(1,28*28);
		auto output = forward<float>(
				X_test[i],
				W1,
				B1,
				W2,
				B2
				);

		auto max_index = output.softmax().argmax()[1];
		if(max_index == Y_test[i].argmax()[1])
			success++;
	}

	float accuracy = static_cast<float>(success) / 
			static_cast<float>(Y_test.size());

	std::cout << "accuracy: " << accuracy << std::endl;

}

int main(){
	mnist();

	return 0;
}
