#include<string>
#include<iostream>

#include"tensor_lib.hpp"

int main(){
	Tensor<double> t3 = tensor::from_list<double,2>({
		{1, 2, 3},
		{15, 44, 3},
		{31, 12, 3},
		{31, 12, 3},
	}, true);

	Tensor<double> t2 = tensor::from_list<double,2>({
		{1, 5, 2},
		{1, 4, 16},
		{1, 10, 32},
		{31, 12, 3},
	}, true);

	//std::cout << t3 << std::endl;
	//std::cout << t2 << std::endl;
	
	Tensor<double> t2tok = tensor::from_list<double,2>({
		{1, 1, 1, 31},
		{5, 4, 10, 12},
		{2, 16, 32, 3}
	}, true);


	//transpose/
	
	auto t2tt = t2.transpose();

	std::cout << t2tok << std::endl;
	std::cout << t2tt << std::endl;

	Tensor<bool> oh = tensor::one_hot(t2);

	std::cout << oh << std::endl;

	Tensor<int> ta1 = tensor::arange<int>(0, 10, 1);
	Tensor<int> ta2 = tensor::arange<int>(10, 20, 4);

	Tensor<int> oh1 = tensor::one_hot(ta1, 3);
	Tensor<bool> oh2 = tensor::one_hot(ta2);

	Tensor<bool> oh22 = ta2.one_hot();


	std::cout << ta1 << std::endl;
	std::cout << oh1 << std::endl;

	std::cout << ta2 << std::endl;
	std::cout << oh2 << std::endl;

	std::cout << "(oh22 == oh2)" << std::endl;
	std::cout << (oh22 == oh2) << std::endl;

	Tensor<int> cs = oh1.cumsum();
	
	std::cout << cs << std::endl;

	std::cout << cs.dimslice(1,2) - 1 << std::endl;



	/*
	std::cout << "t2ok" << std::endl;
	for(auto it = t2tok.begin(); it != t2tok.end(); ++it){
		std::cout << *it << std::endl;
	}

	std::cout << "t2t" << std::endl;
	for(auto it = t2tt.begin(); it != t2tt.end(); ++it){
		std::cout << *it << std::endl;
	}

	*/


	/*
	Tensor<double> longt = tensor::from_list<double,1>(
		{1, 2, 3, 4, 5, 6, 7, 8}, true);

	auto longtr = longt.reshape(4, 2);

	auto dscd = t3.dimslices(1, 1, 2);

	//auto mulres = longtr * dscd;

	//dscd.transpose_();

	auto dscdt = dscd.transpose(0,1);

	std::cout << dscd << std::endl;
	std::cout << dscdt << std::endl;

	auto mmres = tensor::matmul(dscdt, longtr);

	//mulres.backward();
	mmres.backward();

	std::cout << mmres.grad() << std::endl;
	std::cout << dscdt.grad() << std::endl;
	std::cout << dscd.grad() << std::endl;
	std::cout << longtr.grad() << std::endl;
	std::cout << longt.grad() << std::endl;
	std::cout << t3.grad() << std::endl;

	Tensor<double> rnd = tensor::random_normal(0.0, 1.0, 3, 5);
	std::cout << rnd << std::endl;

	Tensor<double> rndcd = tensor::random_normal(0.0, 1.0, rnd);
	std::cout << rndcd << std::endl;

	Tensor<float> rndber = tensor::random_bernoulli<float>(0.5, 10, 10);
	std::cout << rndber << std::endl;

	Tensor<int> eye = tensor::eye<int>(5);
	std::cout << eye << std::endl;

	Tensor<double> rnduni = tensor::random_uniform<double>(-1,
							1, 2, 2);
	std::cout << rnduni << std::endl;

	std::cout << rnduni.softmax() << std::endl;
	std::cout << "sum: " << rnduni.softmax().sum() << std::endl;

	Tensor<std::size_t> rndmn = tensor::random_multinomial<std::size_t, double>(
			rnduni.softmax(), 20);

	std::cout << rndmn << std::endl;





	
	//std::cout << mmres << std::endl;
	*/



	return 0;
}

