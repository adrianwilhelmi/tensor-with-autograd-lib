#include<iostream>

#include"tensor_lib.hpp"

int main(){
	/*
	Tensor<double> t3 = tensor::from_list<double,2>({
		{1, 2, 3},
		{15, 44, 3},
		{31, 12, 3},
		{4, 4, 2}
	}, false);

	Tensor<double> t2 = tensor::from_list<double,2>({
		{1, 5, 2},
		{1, 4, 16},
		{1, 10, 32},
		{4, 4, 64}
	}, true);

	Tensor<double> t = tensor::concat(t2, t3, 0);

	std::cout << t << std::endl;

	Tensor<double> temp = t.dimslices_arange(0, 4, 7);

	Tensor<double> ten = temp * t2;

	std::cout << ten << std::endl;

	Tensor<double> t4 = ten * t3;

	std::cout << t4 << std::endl;

	Tensor<double> t5 = tensor::from_list<double,2>({
		{14, 15, 16},
		{17, 18, 19},
		{21, 22, 23},
		{24, 25, 26}
	}, true);

	Tensor<double> t6 = t4 - t5;

	Tensor<double> t7 = tensor::from_list<double,2>({
		{20, 20, 20},
		{2, 500, 2},
		{2, 500, 2},
		{2, 2, 2}
	}, true);

	Tensor<double> t8 = t6 / t7;

	std::cout << t8 << std::endl;

	Tensor<double> t9 = t8.dimslices(0, 1, 2).dimslice(1, 1);

	std::cout << t9 << std::endl;

	Tensor<double> t10 = t9.tanh();

	t10.backward();

	std::cout << t10.grad() << std::endl;
	std::cout << t9.grad() << std::endl;
	std::cout << t8.grad() << std::endl;
	std::cout << t7.grad() << std::endl;
	std::cout << t6.grad() << std::endl;
	std::cout << t5.grad() << std::endl;
	std::cout << t2.grad() << std::endl;
	std::cout << t.grad() << std::endl;
	*/

	Tensor<double> tensor = tensor::from_list<double,2>({
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{10, 11, 12}
	}, false);

	std::cout << tensor << std::endl;
	tensor.transpose_();
	std::cout << tensor << std::endl;
	std::cout << tensor.transpose() << std::endl;

	Tensor<double> tensor2 = tensor::from_list<double,2>({
		{1, 2, 3},
		{4, 5, 6},
		{10, 11, 12}
	}, false);


	std::cout << tensor2 << std::endl;
	tensor2.diag().tanh_();
	std::cout << tensor2 << std::endl;

	return 0;
}

