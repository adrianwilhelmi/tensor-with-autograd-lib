#ifndef TENSOR_OPS_HPP_
#define TENSOR_OPS_HPP_

#include"storage.hpp"
#include"tensor.hpp"
#include"utils/tensor_utils.hpp"

namespace tensor{
	template<typename T>
	bool same_storage(const Tensor<T>&t1, const Tensor<T>&t2){
		return t1.data() == t2.data();
	}

	template<typename T, std::size_t N>
	Tensor<T> from_list(const TensorInitializer<T,N>& init){
		TensorSlice d;
		tensor_impl::derive_extents<T,N>(init, d);
		d.compute_strides();
		Storage<T> elems(d.size);
		tensor_impl::fill_data<T,N>(init, elems, d, 0, 0);
		return{d, elems};
	}

}; //namespace tensor

#endif //TENSOR_OPS_OPP_


