#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include"storage.hpp"
#include"utils/tensor_slice.hpp"
#include"utils/tensor_utils.hpp"

template<typename T>
class Tensor{
public:


private:
	tensor_slice desc_;
	Storage<T> elems_;
	Storage<T> grads_;

	bool req_grad;
};


#endif //TENSOR_HPP_

