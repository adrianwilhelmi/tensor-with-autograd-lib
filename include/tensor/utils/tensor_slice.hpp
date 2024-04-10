#ifndef TENSOR_SLICE_HPP_
#define TENSOR_SLICE_HPP_

#include<numeric>
#include<type_traits>
#include<cassert>
#include<cstddef>
#include<algorithm>
#include<iostream>

#include"tensor_utils.hpp"
#include"../storage.hpp"

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
	Storage<std::size_t>*extents;
	Storage<std::size_t>*strides;
};

tensor_slice::tensor_slice() : dims{1}, size{1}, start{0} {
	extents = new std::size_t[1]{0};
	strides = new std::size_t[1]{1};
}

tensor_slice::tensor_slice(std::size_t d, std::size_t s,
		std::initializer_list<std::size_t> exts)
	: dims(d), start(s){
		assert(d == exts.size());

		this->extents = new std::size_t[d];
		this->strides = new std::size_t[d];

		std::copy(exts.begin(), exts.end(), this->extents);

		this->size = tensor_impl::compute_strides(d, this->extent, this->strides);
}

tensor_slice::tensor_slice(std::size_t d, std::size_t s,
		std::initializer_list<std::size_t> exts,
		std::initializer_list<std::size_t> strs)
	: dims(d), start(s){
	assert(d == exts.size());

	this->size = tensor_impl::compute_size(d, this->extents);
	this->extents = new std::size_t[d];
	this->strides = new std::size_t[d];
z
	std::copy(exts.begin(), exts.end(), this->extents);
	std::copy(strs.begin(), strs.end(), this->strides);
}

tensor_slice::tensor_slice(std::size_t d, std::vector<std::size_t> exts)
	: dims(d), start(0){
	assert(d == exts.size());
Dims
	this->extents = new std::size_t[d];
	this->strides = new std::size_t[d];

	std::copy(exts.begin(), exts.end(), this->extents);

	this->size = tensor_impl::compute_strides(d, this->extent, this->strides);
}

template<typename... Dims>
tensor_slice::tensor_slice(Dims... exts) : start{0}{
	this->dims = sizeof(exts);

	std::size_t args[this->dims] {std::size_t(exts)...};
	this->extents = args;

	this->strides = new std::size_t[d];

	size = tensor_impl::compute_strides(this->extents, this->strides):
}

template<typename... Dims>
std::size_t tensor_slice::operator()(Dims... exts) const{
	static_assert(sizeof...(exts) == this->dims, "tensor slice(): inconsistent dimensions");
	std::size_t args[this->dims] {std::size_t(dims)...};
	return start + std::inner_product(args, args + this->dims, 
			strides.begin(), std::size_t(0));
}

std::size_t tensor_slice::offset(const Storage<std::size_t>&pos) const{
	assert(pos.size() == );
	return start + 

#endif
