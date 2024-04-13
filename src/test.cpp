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

	std::cout << t3 << std::endl;
	Tensor<int,1> t4 = t3.dimslice(1,1);

	std::cout << "t4(0) = " << t4(0) << std::endl;
	std::cout << "t4(1) = " << t4(1) << std::endl;
	std::cout << "t4(2) = " << t4(2) << std::endl;
	std::cout << "t4(3) = " << t4(3) << std::endl;

	std::cout << "t4" << std::endl;
	std::cout << t4 << std::endl;

	for(auto it = t4.begin(); it != t4.end(); ++it){
		std::cout << *it << std::endl;  
	}

	return 0;
}


