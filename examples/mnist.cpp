#include<fstream>
#include<string>
#include<iostream>
#include<algorithm>
#include<vector>
#include<stdexcept>

#include"tensor_lib.hpp"

template<typename T = float>
std::pair<Tensor<float>, Tensor<float>> csv_to_tensors(
		const std::string& file_path,
		std::size_t num_imgs){
	std::ifstream file(file_path);
	if(!file.is_open())
		throw std::runtime_error("failed to open file: " + file_path);

	Tensor<float> imgs(num_imgs, 28, 28);
	Tensor<float> Y(num_imgs, 1, 10);

	std::string row;
	std::getline(file, row);

	std::size_t count = 0;
	while(std::getline(file, row) && count < num_imgs){
		std::istringstream ss(row);
		std::string token;
		Tensor<T> img(28,28);
		img.enable_grad();
		std::size_t j = 0;

		while(std::getline(ss, token, ',')){
			if(j == 0){
				Tensor<int> temp = std::stoi(token);
				Tensor<float> t = temp.one_hot<float>(10);

				Y[count] += t;
			}
			else{
				img.data()[(j - 1)] = std::stoi(token) / 255.0;
			}
			++j;
		}

		imgs[count] += img;
		++count;
	}
	
	return std::make_pair(std::move(imgs), std::move(Y));
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

int main(){
	//load imgs into tensors

	const std::string train_path = "./examples/dataset/mnist_train.csv";
	const std::string test_path = "./examples/dataset/mnist_test.csv";

	const std::size_t num_imgs = 30000;

	Tensor<Tensor<float>> dataset(num_imgs, 2);

	auto [X, Y] = csv_to_tensors<float>(train_path, 
						num_imgs);
	auto [X_test, Y_test] = csv_to_tensors<float>(test_path, 
							num_imgs);

	dataset(0, 0) = X;
	dataset(0, 1) = Y;


	//initialize net params

	const std::size_t input_size = 28 * 28;
	const std::size_t hidden_size = 128;
	const std::size_t num_classes = 10;

	Tensor<float> W1 = tensor::random_normal<float>(
			0, 0.01, input_size, hidden_size);
	Tensor<float> B1 = tensor::zeros<float>(1, hidden_size);
	Tensor<float> W2 = tensor::random_normal<float>(
			0, 0.01, hidden_size, num_classes);
	Tensor<float> B2 = tensor::zeros<float>(1, num_classes);

	W1.enable_grad();
	B1.enable_grad();
	W2.enable_grad();
	B2.enable_grad();

	const std::size_t num_epochs = 30;
	float learning_rate = 0.01;



	const std::size_t batch_size = 56;
	const std::size_t num_batches = num_imgs / batch_size;

	//learning
	
	auto start_global = std::chrono::high_resolution_clock::now();
	auto start = std::chrono::high_resolution_clock::now();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	for(std::size_t epoch = 0; epoch < num_epochs; ++epoch){
		std::cout << "epoch: " << epoch <<  std::endl; 


		start = std::chrono::high_resolution_clock::now();
		
		if(epoch == 120)
			learning_rate /= 10.0;



		for(std::size_t i = 0; i < num_batches; i += batch_size){
			X.shuffle_(Y);

			std::size_t start_index = tensor::randint(0,
					num_imgs - batch_size - 1,
					1).item();

			Tensor<float> X_batch = X.dimslices_arange(0,
					start_index, 
					start_index + batch_size - 1).reshape(
						batch_size,
						28 * 28);

			Tensor<float> Y_batch = Y.dimslices_arange(0,
					start_index,
					start_index + batch_size - 1).reshape(
						batch_size,
						num_classes);

			auto output = forward<float>(
					X_batch,
					W1,
					B1,
					W2,
					B2
					);


			auto losses = tensor::cross_entropy<float>(output, 
								Y_batch);

			auto loss = tensor::mean(losses);
			loss.backward();
			std::cout << "loss: " << loss;

			W2 -= learning_rate * W2.grad();
			B2 -= learning_rate * B2.grad();
			W1 -= learning_rate * W1.grad();
			B1 -= learning_rate * B1.grad();

			W2.zero_grad();
			B2.zero_grad();
			W1.zero_grad();
			B1.zero_grad();
		}




		/*
		for(std::size_t i = 0; i < num_imgs; ++i){
			Tensor<float> X_batch = X[i].reshape(1,28*28);
			Tensor<float> Y_batch = Y[i].reshape(1,10);

			auto output = forward<float>(
					X_batch,
					W1,
					B1,
					W2,
					B2
					);


			//auto output_reshaped = output.reshape(1,1,10);

			auto loss = tensor::cross_entropy<float>(
							output, 
							Y_batch);

			loss.backward();
			std::cout << "loss: " << loss;

			W2 -= learning_rate * W2.grad();
			B2 -= learning_rate * B2.grad();
			W1 -= learning_rate * W1.grad();
			B1 -= learning_rate * B1.grad();

			W2.zero_grad();
			B2.zero_grad();
			W1.zero_grad();
			B1.zero_grad();
		}
		*/


		stop = std::chrono::high_resolution_clock::now();

		duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
		std::cout << "duration: " << duration.count() << std::endl;
	}



	auto stop_global = std::chrono::high_resolution_clock::now();
	auto duration_global = std::chrono::duration_cast<std::chrono::milliseconds>(stop_global - start_global);

		std::cout << "global duration: " << duration_global.count() << std::endl;


	//testing
	
	std::size_t success = 0;

	std::size_t num_test_imgs = num_imgs / 10;

	for(std::size_t i = 0; i < num_test_imgs; ++i){
		if(i % 500 == 0) std::cout << "test img: " << i << std::endl;
		Tensor<float> X_flat = X_test[i].reshape(1,28*28);
		auto output = forward<float>(
				X_flat,
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
			static_cast<float>(num_test_imgs);

	std::cout << "accuracy: " << accuracy << std::endl;

	return 0;
}

