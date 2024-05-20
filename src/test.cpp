#include<string>
#include<iostream>

#include"tensor_lib.hpp"

int main(){
	Tensor<float> t3 = tensor::from_list<float,2>({
		{1, 2, 3},
		{15, 44, 3},
		{31, 12, 3},
		{31, 12, 3},
	}, true);

	Tensor<float> t2 = tensor::from_list<float,2>({
		{1, 5, 2},
		{1, 4, 16},
		{1, 10, 32},
		{31, 12, 3},
	}, true);

	//Tensor<float> tr1 = tensor::random_normal(0.0, 1.0, 4000, 4000);
	//Tensor<float> tr2 = tensor::random_normal(0.0, 1.0, 4000, 4000);


	/*
	const std::size_t input_size = 8;
	const std::size_t hidden_size = 4;
	const std::size_t num_classes = 2;

	Tensor<float> X = tensor::random_normal<float>(
			0, 0.01, 2, 8);


	Tensor<float> Y = tensor::random_normal<float>(
			0, 0.01, 2, 2);

	Tensor<float> W1 = tensor::random_normal<float>(
			0, 0.01, input_size, hidden_size);
	Tensor<float> B1 = tensor::ones<float>(1, hidden_size);
	Tensor<float> W2 = tensor::random_normal<float>(
			0, 0.01, hidden_size, num_classes);
	Tensor<float> B2 = tensor::ones<float>(1, num_classes);


	std::cout << W1 << std::endl;
	std::cout << B1 << std::endl;
	std::cout << W2 << std::endl;
	std::cout << B2 << std::endl;

	std::cout << "X" << std::endl;
	std::cout << X << std::endl;
	std::cout << X.descriptor() << std::endl;

	std::cout << "Y" << std::endl;
	std::cout << Y << std::endl;
	std::cout << Y.descriptor() << std::endl;

	

	auto hmm = tensor::matmul(X, W1);
	auto hmmb = hmm + B1;
	auto hidden = hmmb.relu();

	std::cout << hmm << std::endl;
	std::cout << hmmb << std::endl;
	std::cout << hidden << std::endl;

	auto omm = tensor::matmul(hidden, W2);
	auto ommb = omm + B2;
	
	std::cout << omm << std::endl;

	std::cout << B2 << std::endl;
	std::cout << ommb << std::endl;

	std::cout << B2.descriptor() << std::endl;
	std::cout << ommb.descriptor() << std::endl;
	*/

	auto ri = tensor::randint(0, 10, 5, 2);
	auto r2 = tensor::randint(0, 10, 5, 2);

	std::cout << ri << std::endl;
	//std::cout << r2 << std::endl;

	ri.shuffle_();

	std::cout << ri << std::endl;




	/*
	auto start = std::chrono::high_resolution_clock::now();

	//std::cout << tr1.matmul_optimized(tr2) << std::endl;
	auto optim_result = tr1.matmul_optimized(tr2);

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	std::cout << "optim: " << duration.count() << std::endl;

	start = std::chrono::high_resolution_clock::now();

	auto classic_result = tensor::matmul_classic(tr1, tr2);

	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	std::cout << "clasic: " << duration.count() << std::endl;




	start = std::chrono::high_resolution_clock::now();

	auto matmul_result = tensor::matmul(tr1, tr2);

	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	std::cout << "matmul: " << duration.count() << std::endl;





	std::cout << "equality" << std::endl;

     
	bool eq = tensor::nearly_equal(optim_result, classic_result);
	bool eq2 = tensor::nearly_equal(optim_result, matmul_result);
	std::cout << eq << std::endl;
	std::cout << eq2 << std::endl;
	

	*/

	return 0;
}

