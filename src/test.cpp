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

	const std::string face = "./test/photos/trll.jpeg";

	auto facet = tensor::from_image<float>(face);

	std::cout << facet.size() << std::endl;
	std::cout << facet.descriptor() << std::endl;

	Tensor<float> kernel2d = tensor::from_list<float,2>({
			/*
			{1/16, 2/16, 1/16},
			{2/16, 4/16, 2/16},
			{1/16, 2/16, 1/16},
			*/

			{-1.0, -1.0, -1.0},
			{-1.0, 8.0, -1.0},
			{-1.0, -1.0, -1.0},
		}, false);

	Tensor<float> kernel3d(3,3,3);
	kernel3d.dimslice(0,0) += kernel2d;
	kernel3d.dimslice(0,1) += kernel2d;
	kernel3d.dimslice(0,2) += kernel2d;

	auto facetc = tensor::conv2d(facet, kernel3d);

	std::cout << facetc.size() << std::endl;
	std::cout << facetc.descriptor() << std::endl;

	const std::string faceconvd = "./src/faceconvd.jpeg";

	tensor::to_image<float>(facetc, faceconvd);

	return 0;
}

