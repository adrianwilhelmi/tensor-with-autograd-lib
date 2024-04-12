#ifndef TENSOR_SLICE_HPP_
#define TENSOR_SLICE_HPP_

#include"../storage.hpp"
#include"../declarations.hpp"
#include"../traits.hpp"
#include"tensor_utils.hpp"


#include<array>
#include<numeric>
#include<type_traits>
#include<cassert>
#include<cstddef>
#include<algorithm>
#include<iostream>


struct tensor_slice{
	tensor_slice();
	tensor_slice(std::size_t s, std::initializer_list<std::size_t> exts);
	tensor_slice(std::size_t s, std::initializer_list<std::size_t> exts, std::initializer_list<std::size_t> strs);

	tensor_slice(Storage<std::size_t> &exts);

	template<typename... Dims>
	tensor_slice(Dims... dims);

	template<typename... Dims>
	std::size_t operator()(Dims... dims) const;

	std::size_t offset(Storage<std::size_t> &pos) const;

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
	: dims(exts.size()), start(s), extents(dims), strides(dims){
		std::copy(exts.begin(), exts.end(), this->extents.begin());

		this->size = tensor_impl::compute_strides(this->extents, 
				this->strides);
}

tensor_slice::tensor_slice(std::size_t s,
		std::initializer_list<std::size_t> exts,
		std::initializer_list<std::size_t> strs)
	: dims(exts.size()), start(s), extents(dims), strides(dims){
	assert(exts.size() == strs.size());

	this->size = tensor_impl::compute_size(this->extents);

	std::copy(exts.begin(), exts.end(), this->extents.begin());
	std::copy(strs.begin(), strs.end(), this->strides.begin());
}

tensor_slice::tensor_slice(Storage<std::size_t>&exts)
	: dims(exts.size()), start(0), extents(dims), strides(dims){
	this->size = tensor_impl::compute_strides(this->extents, this->strides);
}

template<typename... Dims>
tensor_slice::tensor_slice(Dims... exts) : start{0}{
	this->dims = sizeof...(exts);

	std::size_t args[this->dims] {std::size_t(exts)...};
	this->extents = args;

	this->strides(this->dims);
	this->size = tensor_impl::compute_strides(this->extents, this->strides);
}

template<typename... Dims>
std::size_t tensor_slice::operator()(Dims... exts) const{
	static_assert(sizeof...(exts) == this->dims, "tensor slice(): inconsistent dimensions");
	std::size_t args[this->dims] {std::size_t(exts)...};
	return start + std::inner_product(args, args + this->dims, 
			strides.begin(), std::size_t(0));
}

std::size_t tensor_slice::offset(Storage<std::size_t>&pos) const{
	assert(pos.size() == this->dims);
	return start + std::inner_product(pos.begin(), pos.end(),
			strides.begin(), std::size_t{0});
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
	return os;
}

#endif //TENSOR_SLICE_HPP_
