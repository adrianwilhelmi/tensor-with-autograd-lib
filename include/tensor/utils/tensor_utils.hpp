#ifndef TENSOR_UTILS_HPP_
#define TENSOR_UTILS_HPP_

#include<cstddef>
#include<algorithm>
#include<initializer_list>
#include<cmath>

#include"tensor_declarations.hpp"

#include"tensor_slice.hpp"


namespace tensor_impl{
	inline std::size_t compute_strides(const std::size_t d,
			const std::size_t*exts,	std::size_t*strs){
		std::size_t st = 1;
		for(long long i = d - 1; i >= 0; --i){
			strs[i] = st;
			st *= exts[i];
		}
		return st;
	}
	
	inline std::size_t compute_size(std::size_t dims, 
			std::size_t*exts){
		return std::accumulate(exts, exts + dims, 1,
				std::multiplies<std::size_t>{});
	}
};

#endif

