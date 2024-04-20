#include<iostream>

#include"tensor_lib.hpp"

int main(){
	Tensor<int> t3 = tensor::from_list<int,2>({
		{1, 2, 3},
		{15, 44, 3},
		{31, 12, 3},
		{4, 4, 2}
	});

	t3.enable_grad();

	std::cout << t3 << std::endl;

	Tensor<int> t2 = t3.dimslice(1,2);
	
	std::cout << t2 << std::endl;

	std::cout << t3.grad() << std::endl;

	std::cout << "hey" << std::endl;

	t2.backward();

	std::cout << t2.grad() << std::endl;
	std::cout << t3.grad() << std::endl;

	return 0;
}

