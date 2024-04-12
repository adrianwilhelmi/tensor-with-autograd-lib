#ifndef TENSOR_REF_HPP_
#define TENSOR_REF_HPP_

#include<cassert>
#include<cstddef>
#include<array>
#include<iterator>
#include<string>
#include<type_traits>

#include"declarations.hpp"
#include"traits.hpp"
#include"utils/tensor_slice.hpp"
#include"utils/tensor_utils.hpp"

template<typename T, std::size_t N>
class TensorRef{

private:
	T*data;
	std::reference_wrapper(Tensor<T,N>*og);
	
};


#endif
