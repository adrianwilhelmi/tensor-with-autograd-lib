#ifndef TENSOR_ITERATOR_HPP_
#define TENSOR_ITERATOR_HPP_

#include"declarations.hpp"
#include"tensor.hpp"

template<typename T, std::size_t N>
class TensorIterator;

template<typename T, std::size_t N>
bool operator<(const TensorIterator<T,N>&lhs, const TensorIterator<T,N>&rhs);

template<typename T, std::size_t N>
class TensorIterator{
	template<typename U, size_t NN>
	friend std::ostream&operator<<(std::ostream&os, 
			const TensorIterator<U,NN>&iter);

public:
	//using iterator_category = std::forward_iterator_tag;
	using iterator_category = std::random_access_iterator_tag;
	using value_type = typename std::remove_const<T>::type;
	using pointer = T*;
	using reference = T&;
	using difference_type = std::ptrdiff_t;

	TensorIterator(const TensorIterator&) = default;

	TensorIterator(const TensorSlice<N>&s, T*base, bool limit=false);
	TensorIterator&operator=(const TensorIterator&);

	const TensorSlice<N>&descriptor() const {return desc;}

	T&operator*() {return*ptr;}
	T*operator->(){return ptr;}

	const T&operator*() const{return*ptr;}
	const T*operator->() const{return ptr;}

	TensorIterator&operator++();
	TensorIterator operator++(int);

	TensorIterator&operator--();
	TensorIterator operator--(int);

	TensorIterator<T,N>&operator+=(difference_type n);

	friend bool operator< <>(const TensorIterator<T,N>&lhs, const TensorIterator<T,N>&rhs);


	static std::ptrdiff_t difference(const TensorIterator&a, const TensorIterator&b){
		
		assert(a.descriptor() == b.descriptor());

		std::ptrdiff_t diff = 0;
		for(std::size_t i = 0; i < N; ++i){
			auto stride_diff = (a.index[i] - b.index[i]) * a.desc.strides[i];
			diff += stride_diff;
		}

		return diff;
	}

private:
	void increment();
	void decrement();

	T*end;
	T*begin;

	std::array<size_t, N> index;
	const TensorSlice<N>&desc;
	T*ptr;
};

template<typename T, std::size_t N>
TensorIterator<T,N>::TensorIterator(const TensorSlice<N>&s, T*base, bool limit)
	: desc(s) {
	std::fill(index.begin(), index.end(), 0);

	index[0] = desc.extents[0];

	this->end = base + desc.offset(index);
	this->begin = base + s.start;

	if(limit){
		ptr = this->end;
	}
	else{
		ptr = this->begin;
	}
}

template<typename T, std::size_t N>
TensorIterator<T,N>&TensorIterator<T,N>::operator=(const TensorIterator &iter){
	std::copy(iter.index.begin(), iter.index.end(), index.begin());
	ptr = iter.ptr;
	return*this;
}

template<typename T, std::size_t N>
TensorIterator<T,N>&TensorIterator<T,N>::operator++(){
	increment();
	return*this;
}

template<typename T, std::size_t N>
TensorIterator<T,N> TensorIterator<T,N>::operator++(int){
	TensorIterator<T,N> x = *this;
	increment();
	return*x;
}

template<typename T, std::size_t N>
TensorIterator<T,N>&TensorIterator<T,N>::operator--(){
	decrement();
	return*this;
}

template<typename T, std::size_t N>
TensorIterator<T,N> TensorIterator<T,N>::operator--(int){
	TensorIterator<T,N> x = *this;
	decrement();
	return*x;
}

template<typename T, std::size_t N>
inline std::ptrdiff_t operator-(const TensorIterator<T,N>&a, const TensorIterator<T,N>&b){
	return TensorIterator<T,N>::difference(a,b);
}

template<typename T, std::size_t N>
TensorIterator<T,N> operator-(const TensorIterator<T,N>&it, int n){
	TensorIterator<T,N> temp = it;
	for(int i = 0; i < std::abs(n); ++i){
		--temp;
	}
	return temp;
}

template<typename T, std::size_t N>
TensorIterator<T,N> operator-(int n, const TensorIterator<T,N>&it){
	return it - n;
}

template<typename T, std::size_t N>
TensorIterator<T,N> operator+(const TensorIterator<T,N>&it, int n){
	TensorIterator<T,N> temp = it;

	for(int i = 0; i < std::abs(n); ++i){
		if(n > 0) ++temp;
		else --temp;
	}

	return temp;
}

template<typename T, std::size_t N>
TensorIterator<T,N> operator+(int n, TensorIterator<T,N>&it){
	return it + n;
}

template<typename T, std::size_t N>
void TensorIterator<T,N>::increment(){
	if(ptr == end){
		return;
	}

	std::size_t d = N - 1;
	while(true){
		ptr += desc.strides[d];
		++index[d];

		if(index[d] != desc.extents[d]) break;

		if(d != 0){
			ptr -= desc.strides[d] * desc.extents[d];
			index[d] = 0;
			--d;
		}
		else{
			break;
		};
	}
}

template<typename T, std::size_t N>
void TensorIterator<T,N>::decrement(){
	if(ptr == begin){
		return;
	}

	std::size_t d = N - 1;
	while (true) {
		if (index[d] == 0) {
			if (d != 0) {
				ptr -= desc.strides[d] * index[d]; 
				index[d] = desc.extents[d] - 1;
				ptr += desc.strides[d] * index[d];
				--d;
			}
			else{
				break;
			}
		}
		else{
			ptr -= desc.strides[d];
			--index[d];
			break;
		}
	}

	if (ptr < begin) {
		ptr = begin; 
	}
}

template<typename T, std::size_t N>
TensorIterator<T,N>& TensorIterator<T,N>::operator+=(difference_type n){
	if(n >= 0){
		for(difference_type i = 0; i < n; ++i){
			this->increment();
		}
	} else{
		for(difference_type i = 0; i < n; ++i){
			this->decrement();
		}
	}
	return*this;
}

template<typename T, std::size_t N>
std::ostream&operator<<(std::ostream&os, const TensorIterator<T,N> &iter){
	os << "target = " << *iter.ptr << std::endl;
	os << "index = " << iter.index << std::endl;
	return os;
}

template<typename T, std::size_t N>
inline bool operator==(const TensorIterator<T,N>&a, const TensorIterator<T,N>&b){
	assert(a.descriptor() == b.descriptor());
	return &*a == &*b;
}

template<typename T, std::size_t N>
inline bool operator!=(const TensorIterator<T,N>&a, const TensorIterator<T,N>&b){
	return !(a == b);
}

template<typename T, std::size_t N>
bool operator<(const TensorIterator<T,N>&lhs, const TensorIterator<T,N>&rhs){
	return lhs.ptr < rhs.ptr;
}

template<typename T, std::size_t N>
bool operator>(const TensorIterator<T,N>&lhs, const TensorIterator<T,N>&rhs){
	//return (!(lhs.ptr < rhs.ptr) && (lhs != rhs));
	return !(lhs.ptr < rhs.ptr);
}


#endif //TENSOR_ITERATOR_HPP_
