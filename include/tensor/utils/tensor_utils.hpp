#ifndef TENSOR_UTILS_HPP_
#define TENSOR_UTILS_HPP_

#include<cstddef>
#include<algorithm>
#include<initializer_list>
#include<cmath>

#include"tensor_declarations.hpp"

#include"tensor_slice.hpp"


namespace tensor_impl{
	
	inline std::size_t compute_size(std::size_t dims, 
			std::size_t*exts){
		return std::accumulate(exts, exts + dims, 1,
				std::multiplies<std::size_t>{});
	}
};

#endif

