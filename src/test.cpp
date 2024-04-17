#include<iostream>

#include"tensor_lib.hpp"

int main(){
	/*
	Tensor<int,2> t1 = {
		{1, 2, 3},
		{4, 5, 6}
	};

	Tensor<int,2> t2;
	t2 = t1;

	t2(1,1) = 20;

	std::cout << t1 << std::endl;
	std::cout << t2 << std::endl;
	*/

	Tensor<int> t3 = tensor::from_list<int,2>({
		{1, 2, 3},
		{15, 44, 3},
		{31, 12, 3},
		{4, 4, 2}
	});

	std::cout << t3.descriptor() << std::endl;

	std::cout << t3.view(2, 6) << std::endl;

	std::cout << t3 << std::endl;

	std::cout << t3.dimslices(0, 1, 2) << std::endl;

	Tensor<int> t4 = t3.dimslices(0, 0, 3);
	std::cout << t4 << std::endl;

	std::cout << t3.dimslice(1,1) << std::endl;

	Tensor<int> t5 = t3.dimslice(1, 1);
	std::cout << t5 << std::endl;


	return 0;
}


