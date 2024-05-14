#include<string>
#include<iostream>

#include"tensor_lib.hpp"

int main(){
	Tensor<float> t3 = tensor::from_list<float,2>({
		{1, 2, 3},
		{15, 44, 3},
		{31, 12, 3},
		{31, 12, 3},
	}, true);

	Tensor<float> t2 = tensor::from_list<float,2>({
		{1, 5, 2},
		{1, 4, 16},
		{1, 10, 32},
		{31, 12, 3},
	}, true);


	Tensor<float> tr1 = tensor::random_normal(0.0, 1.0, 4, 15);
	Tensor<float> tr2 = tensor::random_normal(0.0, 1.0, 15, 4);

	auto start = std::chrono::high_resolution_clock::now();

	//std::cout << tr1.matmul_optimized(tr2) << std::endl;
	auto optim_result = tr1.matmul_optimized(tr2);

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	std::cout << "optim: " << duration.count() << std::endl;

	start = std::chrono::high_resolution_clock::now();

	auto classic_result = tensor::matmul_classic(tr1, tr2);

	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	std::cout << "clasic: " << duration.count() << std::endl;




	start = std::chrono::high_resolution_clock::now();

	auto matmul_result = tensor::matmul(tr1, tr2);

	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	std::cout << "matmul: " << duration.count() << std::endl;





	std::cout << "equality" << std::endl;


	/*
	std::cout << optim_result << std::endl;
	std::cout << classic_result << std::endl;
	*/

     
	bool eq = tensor::nearly_equal(optim_result, classic_result);
	bool eq2 = tensor::nearly_equal(optim_result, matmul_result);
	std::cout << eq << std::endl;
	std::cout << eq2 << std::endl;
	



	/*

	auto temp1 = matmul_result.transpose();
	
	auto temp2 = matmul_result;
	temp2.transpose_();

	std::cout << temp2.descriptor() << std::endl;
	std::cout << (temp1 == temp2) << std::endl;

	Tensor<float> ttd(matmul_result.extent(1), matmul_result.extent(0));
	ttd.transpose_();
	auto tit = ttd.begin();
	for(auto it = matmul_result.begin(); it != matmul_result.end(); ++it){
		*tit = *it;
		++tit;
	}
	ttd.transpose_();

	std::cout << temp1 << std::endl;
	std::cout << ttd << std::endl;

	std::cout << temp1.descriptor() << std::endl;
	std::cout << ttd.descriptor() << std::endl;
	

	std::cout << temp1.storage() << std::endl;
	std::cout << ttd.storage() << std::endl;

	auto temp1mm = tensor::matmul(matmul_result, temp1);
	auto ttdmm = tensor::matmul(matmul_result, ttd);

	std::cout << temp1mm << std::endl;
	std::cout << ttdmm << std::endl;

	std::cout << (temp1mm == ttdmm) << std::endl;
	
	*/
	return 0;
}

