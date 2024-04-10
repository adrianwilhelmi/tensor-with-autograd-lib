#include<iostream>

#include"../include/tensor/storage.hpp"
#include"../include/tensor/utils/tensor_slice.hpp"

int main(){
	Storage<int> s = {1, 2, 3};

	Storage<int> copy(s);

	std::cout << s << std::endl;
	std::cout << copy << std::endl;

	std::cout << "s == copy "<< (s == copy) << std::endl;
	std::cout << "same storage " << same_storage(s, copy) << std::endl;

	Storage<int> sharee;
	s.share(sharee);

	std::cout << "s == sharee "<< (s == sharee) << std::endl;
	std::cout << "same storage " << same_storage(s, sharee) << std::endl;

	std::cout << "observers of init storage = " <<
		s.observers() << std::endl;

	if(true){
		Storage<int> sharee3;
		s.share(sharee3);

		std::cout << "s == sharee3 "<< (s == sharee3) << std::endl;
		std::cout << "same storage3 " << same_storage(s, sharee3) << std::endl;

		std::cout << "observers of init storage = " <<
			s.observers() << std::endl;
	}

	std::cout << "observers of init storage = " <<
		s.observers() << std::endl;

	return 0;
}


