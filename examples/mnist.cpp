#include<fstream>
#include<string>
#include<iostream>
#include<algorithm>
#include<vector>
#include<stdexcept>

#include"tensor_lib.hpp"

template<typename T = float>
std::pair<std::vector<Tensor<float>>, std::vector<Tensor<float>>> csv_to_tensors(const std::string& file_path,
		std::size_t num_imgs){
	std::ifstream file(file_path);
	if(!file.is_open())
		throw std::runtime_error("failed to open file: " + file_path);

	std::vector<Tensor<float>> imgs;
	std::vector<Tensor<float>> labels;

	std::string row;
	std::getline(file, row);

	std::size_t count = 0;
	while(std::getline(file, row) && count < num_imgs){
		std::istringstream ss(row);
		std::string token;
		Tensor<T> img(28,28);
		img.enable_grad();
		int j = 0;

		while(std::getline(ss, token, ',')){
			if(j == 0){
				Tensor<int> temp = std::stoi(token);
				Tensor<float> t = temp.one_hot<float>(10);

				labels.push_back(t);
			}
			else{
				img.data()[(j - 1)] = std::stoi(token) / 255.0;
			}
			++j;
		}
		imgs.push_back(std::move(img));
		++count;
	}
	
	return std::make_pair(std::move(imgs), std::move(labels));
}


template<typename T>
Tensor<T> forward(
		Tensor<T>& X,
		Tensor<T>& W1,
		Tensor<T>& B1,
		Tensor<T>& W2,
		Tensor<T>& B2
		){
	Tensor<T> hmm = tensor::matmul(X, W1);
	Tensor<T> hmmb = hmm + B1;
	Tensor<T> hidden = hmmb.relu();

	Tensor<T> omm = tensor::matmul(hidden, W2);
	Tensor<T> res = omm + B2;
	return res;
}

int main(){
	//load imgs into tensors

	const std::string train_path = "./examples/dataset/mnist_train.csv";
	const std::string test_path = "./examples/dataset/mnist_test.csv";

	const std::size_t num_imgs = 500;

	auto [train_img, labels] = csv_to_tensors<float>(train_path, num_imgs);
	auto [test_img, test_labels] = csv_to_tensors<float>(test_path, 
			100 * num_imgs);



	//initialize net params

	const std::size_t input_size = 28 * 28;
	const std::size_t hidden_size = 128;
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

	const std::size_t num_epochs = 50;
	float learning_rate = 0.1;
	



	//learning
	
	for(std::size_t epoch = 0; epoch < num_epochs; ++epoch){
		std::cout << "epoch: " << epoch <<  std::endl; 

		if(epoch == 30)
			learning_rate /= 10.0;

		for(std::size_t i = 0; i < num_imgs; ++i){
			Tensor<float> X_flat = train_img[i].reshape(1,28*28);
			auto output = forward<float>(
					X_flat,
					W1,
					B1,
					W2,
					B2
					);

			auto loss = tensor::cross_entropy<float>(output, 
								labels[i]);
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
	}





	//testing
	
	std::size_t success = 0;

	for(std::size_t i = 0; i < test_img.size(); ++i){
		Tensor<float> X_flat = test_img[i].reshape(1,28*28);
		auto output = forward<float>(
				X_flat,
				W1,
				B1,
				W2,
				B2
				);

		auto max_index = output.softmax().argmax()[1];
		if(max_index == test_labels[i].argmax()[1])
			success++;
	}

	float accuracy = static_cast<float>(success) / static_cast<float>(test_img.size());

	std::cout << "accuracy: " << accuracy << std::endl;

	return 0;
}
