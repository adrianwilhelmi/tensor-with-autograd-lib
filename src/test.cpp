#include<iostream>

#include"tensor_lib.hpp"

int main(){
	Tensor<int,2> t1 = {
		{1, 2, 3},
		{4, 5, 6}
	};

	Tensor<int,2> t2;
	t2 = t1;

	
	return 0;
}


