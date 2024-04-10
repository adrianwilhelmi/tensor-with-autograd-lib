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
	tensor_slice();
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
	Storage<std::size_t> extents;
	Storage<std::size_t> strides;
};

tensor_slice::tensor_slice() : dims{1}, size{1}, start{0} {
	this->extents = {0};
	this->strides = {1};
}

tensor_slice::tensor_slice(std::size_t s,
		std::initializer_list<std::size_t> exts)
	: start(s){
		this->d == exts.size();

		this->extents(d);
		this->strides(d);

		std::copy(exts.begin(), exts.end(), this->extents.begin());

		this->size = tensor_impl::compute_strides(d, 
				this->extents, this->strides);
}

tensor_slice::tensor_slice(std::size_t s,
		std::initializer_list<std::size_t> exts,
		std::initializer_list<std::size_t> strs)
	: start(s){
	assert(exts.size() == strs.size());
	this->dims == exts.size();

	this->extents(this->dims);
	this->strides(this->dims);

	this->size = tensor_impl::compute_size(d, this->extents);
z
	std::copy(exts.begin(), exts.end(), this->extents.begin());
	std::copy(strs.begin(), strs.end(), this->strides.begin());
}

tensor_slice::tensor_slice(Storage<std::size_t> exts)
	: dims(exts.size()), start(0){
	this->extents(exts);
	this->strides(this->dims);

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
	return start + std::inner_product(pos.begin(), pos.end(),
			strides.begin(), std::size_t{0});
}

void tensor_slice::clear(){
	size = 0;
	start = 0;
	extents.fill(0);
	strides.fill(0);
}

bool same_extents(const tensor_slice&a, const tensor_slice&b){
	return a.extents == b.extents;
}

inline bool operator==(const tensor_slice&a, const tensor_slice&b){
	return a.start == b.start && a.extents == b.extents && 
		a.strides == b.strides;
}

inline bool operator!=(const tensor_slice&a, const tensor_slice&b){
	return !(a == b);
}

std::ostream&operator<<(std::ostream&os, const tensor_slice&ts){
	os << "size = " << ts.size << std::endl;
	os << "start = " << ts.start << std::endl;
	os << "extents = " << ts.extents << std::endl;
	os << "strides = " << ts.strides << std::endl;
}

std::ostream&operator<<(std::ostream&os, const

#endif //TENSOR_SLICE_HPP_
