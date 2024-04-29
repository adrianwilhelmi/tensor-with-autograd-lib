#ifndef TENSOR_SLICE_HPP_
#define TENSOR_SLICE_HPP_

#include<vector>
#include<numeric>
#include<cassert>
#include<iostream>

struct TensorSlice{
	TensorSlice() : size(1), start(0) {}

	TensorSlice(std::initializer_list<std::size_t> exts) 
		: start(0), extents(exts){
		compute_strides();
	}

	TensorSlice(std::size_t s, std::initializer_list<std::size_t> exts)
		: start(s), extents(exts){
		compute_strides();
	}

	TensorSlice(std::size_t s, std::initializer_list<std::size_t> exts,
			std::initializer_list<std::size_t> strs)
		: start(s), extents(exts), strides(strs){
		compute_size();
	}

	TensorSlice(const std::vector<std::size_t>&exts)
		: start(0), extents(exts){
		compute_strides();
		compute_size();
	}

	TensorSlice(std::size_t order)
		: size(0), start(0), extents(order), strides(order) {
		if(order == 0){
			size = 1;
			extents = {1};
			strides = {1};
		}
	}

	//explicit TensorSlice(Dims... dims) : extents{static_cast<std::size_t>(dims)...}{
	template<typename... Dims>
	explicit TensorSlice(Dims... dims) : start{0}, extents(sizeof...(Dims)){
		std::size_t args[sizeof...(Dims)] {std::size_t(dims)...};
		std::copy(std::begin(args), std::end(args), extents.begin());
		compute_strides();
	}

	void compute_strides(){
		if(extents.empty()) return;

		strides.resize(extents.size());
		size = 1;
		for(int i = extents.size() - 1; i >= 0; --i){
			strides[i] = size;
			size *= extents[i];
		}
	}

	void compute_size(){
		if(extents.empty()) return;

		size = std::accumulate(extents.begin(), extents.end(), 
				1, std::multiplies<std::size_t>());
	}

	std::size_t offset(const std::vector<std::size_t>&pos) const{
		assert(pos.size() == extents.size());
		return start + 
			std::inner_product(pos.begin(), pos.end(),
					strides.begin(), std::size_t(0));
	}

	template<typename... Indices>
	std::size_t operator()(Indices... indices) const{
		assert(sizeof...(Indices) == extents.size() && "inconsistent dimensions");
		std::vector<std::size_t> idx{
			static_cast<std::size_t>(indices)...};
		return offset(idx);
	}

	std::size_t size;
	std::size_t start;
	std::vector<std::size_t> extents;
	std::vector<std::size_t> strides;
};

bool same_extents(const TensorSlice&a, const TensorSlice&b){
	return a.extents == b.extents;
}

inline bool operator==(const TensorSlice&a, const TensorSlice&b){
	return a.start == b.start && a.extents == b.extents &&
		a.strides == b.strides;
}

inline bool operator!=(const TensorSlice&a, const TensorSlice&b){
	return !(a==b);
}

std::ostream&operator<<(std::ostream&os, const TensorSlice&s){
	os << "size = " << s.size << std::endl;
	os << "start = " << s.start << std::endl;
	os << "extents = ";
	for(auto &e : s.extents) os << e << " ";
	os << "\nstrides = ";
	for(auto &e : s.strides) os << e << " ";
	os << std::endl;
	return os;
}

#endif //TENSOR_SLICE_HPP_
