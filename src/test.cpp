#include<iostream>

#include"tensor_lib.hpp"

int main(){
	Tensor<int,2> t1 = {
		{1, 2, 3},
		{4, 5, 6}
	};

	Tensor<int,2> t2;
	t2 = t1;

	t2(1,1) = 20;

	std::cout << t1 << std::endl;
	std::cout << t2 << std::endl;

	Tensor<int,2> t3 = {
		{1, 2, 3},
		{15, 44, 3},
		{31, 12, 3},
		{4, 4, 2}
	};

	t3.sort_();
	std::cout << t3 << std::endl;

	std::sort(t2.begin(), t2.end());

	std::cout << t2 << std::endl;
	std::cout << t1 << std::endl;
	return 0;
}


