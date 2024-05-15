#ifndef TENSOR_ITERATOR_HPP_
#define TENSOR_ITERATOR_HPP_

#include"declarations.hpp"
#include"tensor.hpp"

template<typename T>
class TensorIterator;

template<typename T>
bool operator<(const TensorIterator<T>&lhs, const TensorIterator<T>&rhs);

template<typename T>
class TensorIterator{
	template<typename U>
	friend std::ostream&operator<<(std::ostream&os, 
			const TensorIterator<U>&iter);

public:
	using iterator_category = std::random_access_iterator_tag;
	using value_type = typename std::remove_const<T>::type;
	using pointer = T*;
	using reference = T&;
	using difference_type = std::ptrdiff_t;

	TensorIterator(const TensorIterator&) = default;

	TensorIterator(const TensorSlice&s, T*base, bool limit=false);
	TensorIterator&operator=(const TensorIterator&);

	const TensorSlice& descriptor() const {return desc;}

	T&operator*() {return*ptr;}
	T*operator->(){return ptr;}

	const T&operator*() const{return*ptr;}
	const T*operator->() const{return ptr;}

	TensorIterator&operator++();
	TensorIterator operator++(int);

	TensorIterator&operator--();
	TensorIterator operator--(int);

	TensorIterator<T>&operator+=(difference_type n);

	friend bool operator< <>(const TensorIterator<T>&lhs, const TensorIterator<T>&rhs);


	static std::ptrdiff_t difference(const TensorIterator&a, const TensorIterator&b){
		
		assert(a.descriptor() == b.descriptor());

		std::ptrdiff_t diff = 0;
		std::size_t ord = std::min(a.descriptor().extents.size(), 
					b.descriptor().extents.size());

		for(std::size_t i = 0; i < ord; ++i){
			auto stride_diff = (a.index[i] - b.index[i]) * a.desc.strides[i];
			diff += stride_diff;
		}

		return diff;
	}

	std::size_t size() const{
		return this->desc.size;
	}

	std::vector<std::size_t> get_index(){
		return this->index;
	}

	const std::vector<std::size_t> get_index() const{
		return this->index;
	}

private:
	void increment();
	void decrement();

	void adjust_pointer_for_end(){
		ptr = begin + desc.start + 1;
		for(std::size_t i = 0; i < desc.extents.size(); ++i){
			ptr += desc.strides[i] * (desc.extents[i] - 1);
		}
	}

	T*end;
	T*begin;

	std::vector<std::size_t> index;
	const TensorSlice& desc;
	T*ptr;
};

template<typename T>
TensorIterator<T>::TensorIterator(const TensorSlice&s, T*base, bool limit)
	: begin(base), index(s.extents.size()), desc(s) {
	std::fill(index.begin(), index.end(), 0);

	if(s.size != 0)
		index[0] = desc.extents[0];


	this->end = base + s.offset(index);
	this->begin = base + s.start;

	if(limit){
		//ptr = this->end;

		if(!s.is_contiguous()){
			for(std::size_t i = 0; i < s.extents.size(); ++i){
				index[i] = s.extents[i];
			}
			adjust_pointer_for_end();
		}
		else{
			ptr = this->end;
		}
	}
	else{
		ptr = this->begin;
	}
}

template<typename T>
TensorIterator<T>&TensorIterator<T>::operator=(const TensorIterator &iter){
	std::copy(iter.index.begin(), iter.index.end(), index.begin());
	ptr = iter.ptr;
	end = iter.end;
	begin = iter.begin;
	return*this;
}

template<typename T>
TensorIterator<T>&TensorIterator<T>::operator++(){
	increment();
	return*this;
}

template<typename T>
TensorIterator<T> TensorIterator<T>::operator++(int){
	TensorIterator<T> x = *this;
	increment();
	return*x;
}

template<typename T>
TensorIterator<T>&TensorIterator<T>::operator--(){
	decrement();
	return*this;
}

template<typename T>
TensorIterator<T> TensorIterator<T>::operator--(int){
	TensorIterator<T> x = *this;
	decrement();
	return*x;
}

template<typename T>
inline std::ptrdiff_t operator-(const TensorIterator<T>&a, const TensorIterator<T>&b){
	return TensorIterator<T>::difference(a,b);
}

template<typename T>
TensorIterator<T> operator-(const TensorIterator<T>&it, int n){
	TensorIterator<T> temp = it;
	for(int i = 0; i < std::abs(n); ++i){
		--temp;
	}
	return temp;
}

template<typename T>
TensorIterator<T> operator-(int n, const TensorIterator<T>&it){
	return it - n;
}

template<typename T>
TensorIterator<T> operator+(const TensorIterator<T>&it, int n){
	TensorIterator<T> temp = it;

	for(int i = 0; i < std::abs(n); ++i){
		if(n > 0) ++temp;
		else --temp;
	}

	return temp;
}

template<typename T>
TensorIterator<T> operator+(int n, TensorIterator<T>&it){
	return it + n;
}

template<typename T>
void TensorIterator<T>::increment(){
	/*
	if(ptr == end){
		std::cout << "end " << *ptr << " ";
		return;
	}
	*/

	std::size_t d = index.size() - 1;
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

template<typename T>
void TensorIterator<T>::decrement(){
	/*
	if(ptr == begin){
		return;
	}
	*/

	std::size_t d = index.size() - 1;
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

template<typename T>
TensorIterator<T>& TensorIterator<T>::operator+=(difference_type n){
	if(n >= 0){
		for(difference_type i = 0; i < std::abs(n); ++i){
			this->increment();
		}
	} else{
		for(difference_type i = 0; i < std::abs(n); ++i){
			this->decrement();
		}
	}
	return*this;
}

template<typename T>
std::ostream&operator<<(std::ostream&os, const TensorIterator<T> &iter){
	os << "target = " << *iter.ptr << std::endl;
	os << "index = " << iter.index << std::endl;
	return os;
}

template<typename T>
inline bool operator==(const TensorIterator<T>&a, const TensorIterator<T>&b){
	assert(a.descriptor() == b.descriptor());
	return &*a == &*b;
}

template<typename T>
inline bool operator!=(const TensorIterator<T>&a, const TensorIterator<T>&b){
	return !(a == b);
}

template<typename T>
bool operator<(const TensorIterator<T>&lhs, const TensorIterator<T>&rhs){
	return lhs.ptr < rhs.ptr;
}

template<typename T>
bool operator>(const TensorIterator<T>&lhs, const TensorIterator<T>&rhs){
	//return (!(lhs.ptr < rhs.ptr) && (lhs != rhs));
	return !(lhs.ptr < rhs.ptr);
}

#endif //TENSOR_ITERATOR_HPP_
