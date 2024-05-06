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

	Tensor<double> t3d(2,4,3);

	t3d.dimslice(0,0) += t2;
	t3d.dimslice(0,1) += t2 * t3;

	std::cout << t3d << std::endl;

	t3d.transpose_(1, 2);

	t3d.enable_grad();

	std::cout << t3d << std::endl;

	Tensor<double> twos = t3d.copy_dims();
	twos.fill(2.0);
	twos.enable_grad();

	Tensor<double> t3d2 = t3d * twos;

	std::cout << t3d2 << std::endl;

	Tensor<double> t2d = t3d2.reshape(
			t3d2.extent(0) * t3d2.extent(1),
			t3d2.extent(2)
			);

	std::cout << t2d << std::endl;

	t2d.backward();

	std::cout << t3d.grad() << std::endl;
	std::cout << twos.grad() << std::endl;
	std::cout << t3d << std::endl;






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

