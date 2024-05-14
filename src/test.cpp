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


	Tensor<int> tr1 = tensor::random_normal(0.0, 1.0, 64, 64);
	Tensor<int> tr2 = tensor::random_normal(0.0, 1.0, 64, 64);

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


	/*
	std::cout << optim_result << std::endl;
	std::cout << classic_result << std::endl;
	*/
     
	bool eq = tensor::nearly_equal(optim_result, classic_result);
	bool eq2 = tensor::nearly_equal(optim_result, matmul_result);
	std::cout << eq << std::endl;
	std::cout << eq2 << std::endl;

	return 0;
}

