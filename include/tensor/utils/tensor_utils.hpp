#ifndef TENSOR_UTILS_HPP_
#define TENSOR_UTILS_HPP_

#include<cstddef>
#include<algorithm>
#include<initializer_list>
#include<cmath>
#include<numeric>

#include"../declarations.hpp"
#include"../traits.hpp"
#include"tensor_slice.hpp"

namespace tensor_impl{
	template<std::size_t N, typename List>
	inline bool check_non_jagged(const List& list);

	template<typename T, std::size_t N>
	struct TensorInit{
		using type = std::initializer_list<typename TensorInit<T, N-1>::type>;
	};

	template<typename T>
	struct TensorInit<T,1>{
		using type = std::initializer_list<T>;
	};

	//n == 0 -> err
	template<typename T>
	struct TensorInit<T,0>;
		
	template<std::size_t N, typename I, typename T>
	Enable_if<(N == 1), void> add_extents(I& first,
			const std::initializer_list<T> &list){
		*first = list.size();
	}

	template<std::size_t N, typename I, typename List>
	Enable_if<(N > 1), void> add_extents(I& first, const List& list){
		assert(check_non_jagged<N>(list));
		*first++ = list.size();
		add_extents<N-1>(first, *list.begin());
	}

	template<std::size_t N, typename List>
	std::array<std::size_t,N> inline derive_extents(const List& list){
		std::array<std::size_t, N> a;
		auto f = a.begin();
		add_extents<N>(f,list);
		return a;
	}

	template<std::size_t N, typename List>
	inline bool check_non_jagged(const List& list){
		auto i = list.begin();
		for(auto j = i + 1; j != list.end(); ++j){
			if(derive_extents<N-1>(*i) != derive_extents<N-1>(*j))
				return false;
		}
		return true;
	}

	template<std::size_t N>
	inline std::size_t compute_strides(const std::array<std::size_t, N> &exts,
			std::array<std::size_t, N> &strs){
		std::size_t st = 1;
		for(long long i = N - 1; i >= 0; --i){
			strs[i] = st;
			st *= exts[i];
		}
		return st;
	}

	template<std::size_t N>
	inline std::size_t compute_size(const std::array<size_t, N> &exts){
		return std::accumulate(exts.begin(), exts.end(), 1, 
				std::multiplies<std::size_t>{});
	}

	template<typename T, typename Vec>
	void add_list(const T*first, const T*last, Vec& vec){
		vec.insert(vec.end(), first, last);
	}

	template<typename T, typename Vec>
	void add_list(const std::initializer_list<T>*first, const std::initializer_list<T>*last, Vec& vec){
		for(;first != last; ++first)
			add_list(first->begin(), first->end(), vec);
	}

	template<typename T, typename Vec>
	inline void insert_flat(std::initializer_list<T> list, Vec& vec){
		add_list(list.begin(), list.end(), vec);
	}

	template<typename T, typename Iter>
	inline void copy_list(const T*first, const T*last, Iter&iter){
		iter = std::copy(first, last, iter);
	}

	template<typename T, typename Iter>
	inline void copy_list(const std::initializer_list<T> *first,
			const std::initializer_list<T> *last, Iter &iter){
		for(; first != last; ++first)
			copy_list(first->begin(), first->end(), iter);
	}

	template<typename T, typename Iter>
	inline void copy_flat(std::initializer_list<T> list, Iter&iter){
		copy_list(list.begin(), list.end(), iter);
	}

	template<std::size_t N, typename... Dims>
	inline bool check_bounds(const TensorSlice<N>& slice, Dims... dims){
		static_assert(sizeof...(Dims) == N, "inconsistent dimensions");
		std::size_t indexes[N] {std::size_t(dims)...};
		return std::equal(indexes, indexes + N, slice.extents.begin(),
			       	std::less<std::size_t> {});
	}


	template<typename... Args>
	constexpr bool Requesting_element(){
		return All(Convertible<Args, std::size_t>()...);
	}

	template<typename T>
	inline T dot_product(const TensorRef<T,1>& a, const TensorRef<T,1>& b){
		return std::inner_product(a.begin(), a.end(), b.begin(), T());
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
}; //namespace tensor_impl

template<typename T, std::size_t N>
using TensorInitializer = typename tensor_impl::TensorInit<T,N>::type;

#endif //TENSOR_UTILS_HPP_
