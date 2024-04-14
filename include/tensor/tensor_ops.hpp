#ifndef TENSOR_OPS_HPP_
#define TENSOR_OPS_HPP_

#include"storage.hpp"
#include"tensor.hpp"

namespace tensor{
	template<typename T, std::size_t N>
	bool same_storage(const Tensor<T,N>&t1, const Tensor<T,N>&t2){
		return t1.data() == t2.data();
	}

}; //namespace tensor

#endif //TENSOR_OPS_OPP_


