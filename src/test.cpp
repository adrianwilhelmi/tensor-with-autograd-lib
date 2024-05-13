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


	Tensor<float> tr1 = tensor::random_normal(0.0, 1.0, 17, 32);
	Tensor<float> tr2 = tensor::random_normal(0.0, 1.0, 32, 13);

	std::cout << "STORAGEs" << std::endl;
	std::cout << "A" << std::endl;
	std::cout << tr1.storage() << std::endl;
	std::cout << "B" << std::endl;
	std::cout << tr2.storage() << std::endl;
	std::cout << std::endl;

	std::cout << "og tensors" << std::endl;
	std::cout << "A" << std::endl;
	std::cout << tr1 << std::endl;
	std::cout << "B" << std::endl;
	std::cout << tr2 << std::endl;
	std::cout << std::endl;

	std::cout << "matmuls" << std::endl;
	std::cout << tr1.matmul_optimized(tr2) << std::endl;
	std::cout << tensor::matmul(tr1, tr2) << std::endl;



	/*
	Tensor<float> t81 = tensor::arange(0, 64, 1).reshape(8,8);
	Tensor<float> t82 = tensor::arange(64, 128, 1).reshape(8,8);

	auto mmo = t81.matmul_optimized(t82);
	auto mmc = tensor::matmul(t81,t82);

	std::cout << mmo << std::endl;
	std::cout << mmc << std::endl;
	*/

	

	/*
	auto t2t = t2.transpose();
	auto mmo = t3.matmul_optimized(t2t);
	auto mmc = tensor::matmul(t3,t2t);
	*/



	return 0;
}

