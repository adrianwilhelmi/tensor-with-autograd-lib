#ifndef TENSOR_SLICE_HPP_
#define TENSOR_SLICE_HPP_

#include<numeric>
#include<type_traits>
#include<cassert>
#include<cstddef>
#include<algorithm>
#include<iostream>

#include"tensor_utils.hpp"

struct tensor_slice{
	tensor_slice(); //= default;
	tensor_slice(std::size_t dims, std::size_t s, std::initializer_list<std::size_t> exts);
	tensor_slice(std::size_t dims, std::size_t s, std::initializer_list<std::size_t> exts, std::initializer_list<std::size_t> strs);

	template<typename N>
	tensor_slice(const std::array<std::size_t, N> &exts);

	template<typename... Dims>
	tensor_slice(Dims... dims);

	template<typename... Dims>
	std::size_t operator()(Dims... dims) const;

	template<typename N>
	std::size_t offset(const std::array<std::size_t, N> &pos) const;

	void clear();

	std::size_t dims;
	std::size_t size;
	std::size_t start;
	std::size_t*extents;
	std::size_t*strides;
};

tensor_slice::tensor_slice() : dims{1}, size{1}, start{0} {
	extents = new std::size_t[1]{0};
	strides = new std::size_t[1]{1};
}

tensor_slice::tensor_slice(std::size_t d, std::size_t s,
		std::initializer_list<std::size_t> exts)
	: dims{dims}, start{s}{
		assert(d == exts.size());
		this->extents = new std::size_t[d];
		this->strides = new std::size_t[d];
		std::copy(exts.begin(), exts.end(), this->extents);
		std::copy(strs.begin(), strs.end(), this->strides);
		this->size = tensor_impl::compute_size(d, exts);
	}

#endif
