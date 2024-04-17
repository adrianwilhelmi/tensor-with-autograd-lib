#ifndef TENSOR_UTILS_HPP_
#define TENSOR_UTILS_HPP_

#include<cstddef>
#include<algorithm>
#include<initializer_list>
#include<cmath>
#include<numeric>
#include<type_traits>

#include"../declarations.hpp"
#include"../traits.hpp"
#include"tensor_slice.hpp"

namespace tensor_impl{
	template<typename T, std::size_t N>
	struct TensorInit{
		using type = std::initializer_list<
			typename TensorInit<T,N-1>::type>;
	};

	template<typename T>
	struct TensorInit<T,1>{
		using type = std::initializer_list<T>;
	};

	template<typename T>
	struct TensorInit<T,0>;

	template<typename T>
	bool check_consistency(const std::initializer_list<T>&list){
		return true;
	}

	template<typename List>
	bool check_consistency(const List&list){
		if constexpr (std::is_same_v<typename List::value_type,
				std::initializer_list<
				typename List::value_type::value_type>>) {
			std::size_t first_size = list.size();
			for(const auto&sublist : list){
				if(sublist.size() != first_size ||
					!check_consistency(sublist))
					return false;
			}
		}
		return true;
	}

	template<typename... Dims>
	inline bool check_bounds(const TensorSlice& slice, Dims... dims){
		std::vector<std::size_t> indices{
			static_cast<std::size_t>(dims)...};

		if(indices.size() != slice.extents.size())
			return false;

		return std::equal(indices.begin(), indices.end(), 
			slice.extents.begin(),
			[](std::size_t idx, std::size_t extent){
				return idx < extent;
			});
	}

	template<typename... Args>
	constexpr bool Requesting_element(){
		return All(Convertible<Args, std::size_t>()...);
	}

	template<typename T>
	T relu(const T& x){
		return x > 0 ? x : 0;
	}

	template<typename T>
	T tanh(const T& x){
		return std::tanh(x);
	}

	template<typename T>
	T sigmoid(const T&x){
		return 1.0 / (1 + std::exp(-x));
	}	

	template<typename T, std::size_t N>
	void derive_extents(const typename TensorInit<T,N>::type& init, 
			TensorSlice& s){
		s.extents.push_back(init.size());
		if constexpr(N > 1){
			derive_extents<T,N-1>(*init.begin(), s);
		}
	}

	template<typename T, std::size_t N>
	void fill_data(const typename TensorInit<T,N>::type& init, 
			Storage<T>& data, TensorSlice& slice, 
			std::size_t index = 0, std::size_t offset = 0){
		std::size_t stride = (index < slice.strides.size() - 1) ?
			slice.strides[index] : 1;
		std::size_t n = 0;
		for(const auto&element : init){
			if constexpr(N == 1){
				data[offset + n * stride] = element;
			}
			else{
				fill_data<T,N-1>(element, data, slice, index + 1,
						offset + n * stride);
			}
			++n;
		}
	}
}; //namespace tensor_impl

template<typename T, std::size_t N>
using TensorInitializer = typename tensor_impl::TensorInit<T,N>::type;

#endif //TENSOR_UTILS_HPP_
