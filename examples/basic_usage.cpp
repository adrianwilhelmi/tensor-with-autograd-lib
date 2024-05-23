/*
#include"tensor_lib.hpp"

void basic_usage_example(){

	//Create tensor from initializer list with gradient tracking on.
	//Setting second argument as false will not enable gradient tracking.
	Tensor<float> t1 = tensor::from_list<float,2>({
		{0, 0, 4, 4},
		{2, 2, 13, 14},
		{0, 17, 3, 3}
	}, true);

	//Print tensors on stdout with order up to 3
	std::cout << t1 << std::endl;

	//true
	std::cout << t1.requires_grad() << std::endl;

	//By default all tensors have disabled gradient tracking, 
	//unless it was created by an operation on tensor with 
	//gradient tracking enabled.

	//Create a tensor from initializer list with 
	//gradient tracking disabled
	Tensor<float> t2 = tensor::from_list<float,2>({
		{0.5, 0, 4, -10},
		{2, 2.5, 1, -20.2},
		{0, 170, 3, 3}
	});

	//false
	std::cout << t2.requires_grad() << std::endl;

	//Basic ops are implemented using SIMD operations
	
	Tensor<float> t3 = t1 + t2;

	//true (since t1 requires grad)
	std::cout << t3.requires_grad() << std::endl;

	t2.enable_grad();

	//true
	std::cout << t2.requires_grad() << std::endl;


	//Create tensor with shape/descriptor = (3,4,4)
	Tensor<float> t4 = tensor::random_normal<float>(
					0,	// mean
					0.1,	// std dev
					3,	// 1st extent
					4,	// 2nd extent
					4);	// 3rd extent

	Tensor<float> t42 = tensor::random_normal<float>(
					0,	// mean
					0.1,	// std dev
					t4);	// copy shape from tensor t4


	//Create tensor which is 2nd subtensor of 1st dimension of tensor t4
	Tensor<float> t4s1 = t4.dimslice(0, 1);

	//t4s2 - 1st subtensor of 2nd dimension of tensor t4
	Tensor<float> t4s2 = t4.dimslice(1, 0);

	//dimslice - generalization of row and column up to any dimension
	//dimslice(0, i) = row(i);
	//dimslice(1, i) = col(i);
	
	//dimslices(dim, exts...) - generalization of dimslice to get
	//many subtensors and concatenate them into one
	
	//dimslices_range(dim, i, j) - get i-th, (i+1)-th, ..., j-th 
	//subtensors from (dim) dimension 

	//Print information about tensor's shape (4, 4)
	std::cout << t4s1.descriptor() << std::endl;

	//(3, 4)
	std::cout << t4s2.descriptor() << std::endl;

	//(3, 4, 4)
	std::cout << t4.descriptor() << std::endl;


	auto t5 = t3 * t4s2;

	auto t6 = t5.relu();


	//calculates grad for t6, t5, t3, t2 and t1
	t6.backward();

	//get t3 grads
	std::cout << t3.grad() << std::endl;

	//sets grads to zero
	t5.zero_grad()






}
*/

