#ifndef TENSOR_UTILS_HPP_
#define TENSOR_UTILS_HPP_

#include<cstddef>
#include<algorithm>
#include<initializer_list>
#include<cmath>

#include"tensor_slice.hpp"

namespace tensor_impl{
	inline std::size_t compute_strides(Storage<std::size_t>exts,	
			Storage<std::size_t>strs){
		std::size_t st = 1;
		std::size_t d = exts.size();
		for(long long i = d - 1; i >= 0; --i){
			strs[i] = st;
			st *= exts[i];
		}
		return st;
	}
	
	inline std::size_t compute_size(Storage<std::size_t>&exts){
		return std::accumulate(exts.begin(), exts.end(), 1,
				std::multiplies<std::size_t>{});
	}
}; //namespace tensor_impl

#endif //TENSOR_UTILS_HPP_
