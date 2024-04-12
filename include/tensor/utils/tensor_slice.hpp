#ifndef TENSOR_SLICE_HPP_
#define TENSOR_SLICE_HPP_

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

template<std::size_t N>
struct TensorSlice{
	TensorSlice();
	TensorSlice(std::size_t s, std::initializer_list<std::size_t> exts);
	TensorSlice(std::size_t s, std::initializer_list<std::size_t> exts, std::initializer_list<std::size_t> strs);

	TensorSlice(const std::array<std::size_t,N> &exts);

	template<typename... Dims>
	TensorSlice(Dims... dims);

	template<typename... Dims>
	std::size_t operator()(Dims... dims) const;

	std::size_t offset(const std::array<std::size_t,N> &pos) const;

	std::size_t size;
	std::size_t start;
	std::array<std::size_t,N> extents;
	std::array<std::size_t,N> strides;
};

template<std::size_t N>
TensorSlice<N>::TensorSlice() : size{1}, start{0} {
	std::fill(extents.begin(), extents.end(), 0);
	std::fill(strides.begin(), strides.end(), 1);
}

template<std::size_t N>
TensorSlice<N>::TensorSlice(std::size_t s, std::initializer_list<std::size_t> exts) : start(s) {
	assert(exts.size() == N);
	std::copy(exts.begin(), exts.end(), extents.begin());
	size = tensor_impl::compute_strides(extents, strides);
}

template<std::size_t N>
TensorSlice<N>::TensorSlice(std::size_t s,
		std::initializer_list<std::size_t> exts,
		std::initializer_list<std::size_t> strs)
	: start(s){
	assert(exts.size() == N);
	std::copy(exts.begin(), exts.end(), extents.begin());
	std::copy(strs.begin(), strs.end(), strides.begin());
	size = tensor_impl::compute_size(extents);
}

template<std::size_t N>
TensorSlice<N>::TensorSlice(const std::array<std::size_t, N> &exts)
	: start{0}, extents(exts){
	assert(exts.size() == N);
	size = tensor_impl::compute_strides(extents, strides);
}

template<std::size_t N> 
template<typename... Dims>
TensorSlice<N>::TensorSlice(Dims... dims) : start{0}{
	static_assert(sizeof...(Dims) == N,
			"tensor slice constructor: inconsistent dimensions");

	std::size_t args[N] {std::size_t(dims)...};
	std::copy(std::begin(args), std::end(args), extents.begin());
	size = tensor_impl::compute_strides(extents, strides);
}

template<std::size_t N>
template<typename... Dims>
std::size_t TensorSlice<N>::operator()(Dims... dims) const{
	//static_assert(sizeof...(Dims) == N, "tensor slice (): inconsistent dimensions");

	std::size_t args[N] {std::size_t(dims)...};
	return start + std::inner_product(args, args+N, strides.begin(), std::size_t(0));
}

template<std::size_t N>
std::size_t TensorSlice<N>::offset(const std::array<std::size_t, N> &pos) const{
	assert(pos.size() == N);
	return start + std::inner_product(pos.begin(), pos.end(), strides.begin(), size_t{0});
}

template<std::size_t N>
bool same_extents(const TensorSlice<N>&a, const TensorSlice<N>&b){
	return a.extents == b.extents;
}

template<std::size_t N>
std::ostream&operator<<(std::ostream &os, const std::array<std::size_t, N> &a){
	for(auto x : a) os << x << ' ';
	return os;
}

template<std::size_t N>
inline bool operator==(const TensorSlice<N>&a, const TensorSlice<N>&b){
	return a.start == b.start &&
		std::equal(a.extents.cbegin(), a.extents.cend(), b.extents.cbegin()) && 
		std::equal(a.strides.cbegin(), a.strides.cend(), b.strides.cbegin());
}

template<std::size_t N>
inline bool operator!=(const TensorSlice<N>&a, const TensorSlice<N>&b){
	return !(a == b);
}

template<std::size_t N>
std::ostream&operator<<(std::ostream&os, const TensorSlice<N> &ms){
	os << "size = " << ms.size << std::endl;
	os << "start = " << ms.start << std::endl;
	os << "extents = " << ms.extents << std::endl;
	os << "strides = " << ms.strides << std::endl;
	return os;
}

#endif // TENSOR_SLICE_HPP_

