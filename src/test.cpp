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

	/*
	Tensor<double> tensor = tensor::from_list<double,2>({
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
		{10, 11, 12}
	}, true);

	Tensor<double> t3d(2, 4, 3);
	t3d.enable_grad();

	t3d.dimslice(0,0) += tensor;
	t3d.dimslice(0,1) += tensor * 2.0;

	auto temp1 = t3d.dimslice(0,0);
	auto temp2 = t3d.dimslice(0,1);

	Tensor<double> catted = tensor::concat(temp1,
			temp2, 0);

	std::cout << t3d << std::endl;

	std::cout << catted << std::endl;

	catted.backward();
	
	std::cout << tensor.grad() << std::endl;

	Tensor<double> tensor2 = tensor::from_list<double,2>({
		{1, 2, 3},
		{4, 5, 6},
		{10, 11, 12}
	}, false);

	std::cout << t3d.view(8, 3) << std::endl;

	std::cout << t3d.view(4, 6) << std::endl;

	*/

	Tensor<double> vec = tensor::from_list<double,1>(
		{1, 5, 4, 8, 9, 4}
		, false);

	std::cout << vec << std::endl;

	std::cout << vec.dimslice(0, 1) << std::endl;

	auto s1 = vec.dimslice(0,1);
	auto s2 = vec.dimslice(0,4);

	s1.enable_grad();
	s2.enable_grad();

	std::cout << s1 << std::endl;
	std::cout << s2 << std::endl;

	std::cout << s1 + s2 << std::endl;

	auto s = s1 * s2;

	s.backward();

	std::cout << s1.grad() << std::endl;
	std::cout << s2.grad() << std::endl;

	return 0;
}

