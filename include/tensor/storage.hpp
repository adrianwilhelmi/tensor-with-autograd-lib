#ifndef STORAGE_HPP_
#define STORAGE_HPP_

#include<cassert>
#include<algorithm>
#include<cstddef>
#include<ostream>
#include<iostream>
#include<iomanip>
#include<array>
#include<vector>
#include<memory>

#include"traits.hpp"

template<typename T>
class StorageIterator;

//something between std::array and std::shared_ptr
template<typename T>
class Storage{
public:
	using value_type = T;

	using iterator = StorageIterator<T>;
	using const_iterator = StorageIterator<const T>;

	iterator begin() {return iterator(data_.get());}
	iterator end() {return iterator(data_.get() + size_);}

	const_iterator begin() const {return const_iterator(data_.get());}
	const_iterator end() const {return const_iterator(data_.get() + size_);}

	//constructors
	Storage() : size_(0) {}

	explicit Storage(std::size_t n) 
		: data_(new T[n], std::default_delete<T[]>()), size_(n) {}

	Storage(const Storage& other) : size_(other.size_){
		if(other.data_){
			T*new_data = new T[size_];
			std::copy(other.begin(), other.end(), new_data);
			data_.reset(new_data);
		}
	}

	Storage&operator=(const Storage&other){
		if(this != &other){
			Storage temp(other);
			std::swap(data_, temp.data_);
			size_ = temp.size_;
		}
		return*this;
	}

	Storage(Storage&&other) noexcept
	       : data_(std::move(other.data_)), size_(other.size_){
	       other.size_ = 0;
       }	       

	Storage&operator=(Storage&&other) noexcept{
		if(this != &other){
			data_ = std::move(other.data_);
			size_ = other.size_;
			other.size_ = 0;
		}
		return*this;
	}

	Storage(std::initializer_list<T> list)
		: data_(new T[list.size()], std::default_delete<T[]>()), 
		size_(list.size()){
			std::copy(list.begin(), list.end(), data_.get());
	}

	void share(Storage& other){
		if(this != &other){
			other.data_ = this->data_;
			other.size_ = this->size_;
		}
	}

	//indexing
	T&operator[](std::size_t i){
		return data_[i];
	}

	const T&operator[](std::size_t i) const{
		return data_[i];
	}

	//misc
	std::size_t size(){
		return this->size_;
	}

	T*data(){
		return this->data_.get();
	}

	unsigned int observers() const{
		return data_.use_count();
	}

	template<typename U>
	bool equal(const Storage<U>&other) const{
		if(this->size_ != other.size_) return false;
		return std::equal(this->data_.get(), 
				this->data_.get() + this->size_, other.data_.get());
	}

	template<typename U>
	Storage<U> convert(){
		static_assert(Convertible<U,T>(), "inconsistent types");
		Storage<U> result(size_);
		for(std::size_t i = 0; i < size_; ++i){
			result[i] = static_cast<U>(data_[i]);
		}
		return result;
	}
	
	void fill(const T&val){
		std::fill(this->begin(), this->end(), val);
	}

private:
	std::shared_ptr<T[]> data_;
	std::size_t size_;
};


template<typename T>
inline bool operator==(const Storage<T>&a, const Storage<T>&b){
	return a.equal(b);
}

template<typename T>
inline bool operator!=(const Storage<T> a, const Storage<T>b){
	return !(a == b);
}

template<typename T>
std::ostream&operator<<(std::ostream&os, const Storage<T>&s){
	for(auto it = s.begin(); it != s.end(); ++it){
		os << std::setfill('O') << *it << " ";
	}
	return os;
}

namespace storage{
	template<typename T>
	bool same_storage(Storage<T>&a, Storage<T>&b){
		//assert(same type a, b)
		return a.data() == b.data();
	}

	template<typename T, std::size_t N>
	Storage<T> from_array(std::array<T,N>&arr){
		Storage<T> res(N);
		std::copy(arr.begin(), arr.end(), res.begin());
		return res;
	}

	template<typename T>
	Storage<T> from_vector(std::vector<T>&v){
		Storage<T> res(v.size());
		std::copy(v.begin(), v.end(), res.begin());
		return res;
	}

	template<typename T, std::size_t N>
	std::array<T,N> to_array(Storage<T>&s){
		std::array<T,N> arr;
		std::copy(s.begin(), s.end(), arr.begin());
		return arr;
	}

	template<typename T>
	std::vector<T> to_vector(Storage<T>&s){
		std::vector<T> vec(s.size());
		std::copy(s.begin(), s.end(), vec.begin());
		return vec;
	}
}; //namespace storage

template<typename T>
class StorageIterator{
public:
	using value_type = T;
	using difference_type = std::ptrdiff_t;
	using pointer = T*;
	using reference = T&;
	using iterator_category = std::random_access_iterator_tag;

	StorageIterator(pointer ptr) : ptr{ptr} {}

	reference operator*() const {return *ptr;}
	pointer operator->() {return ptr;}

	StorageIterator&operator++() {++ptr; return*this;}
	StorageIterator operator++(int) {
		StorageIterator temp = *this; 
		++(*this);
		return temp;
	}
	StorageIterator&operator--() {--ptr; return*this;}
	StorageIterator operator--(int) {
		StorageIterator temp = *this; 
		--(*this);
		return temp;
	}

	StorageIterator operator+(difference_type n) const {
		return StorageIterator(ptr + n);
	}
	StorageIterator operator-(difference_type n) const {
		return StorageIterator(ptr - n);
	}
	StorageIterator operator+=(difference_type n) {ptr += n; return*this;}
	StorageIterator operator-=(difference_type n) {ptr -= n; return*this;}

	bool operator==(const StorageIterator&other) const {return ptr == other.ptr;}
	bool operator!=(const StorageIterator&other) const {return ptr != other.ptr;}
	bool operator<(const StorageIterator&other) const {return ptr < other.ptr;}
	bool operator>(const StorageIterator&other) const {return ptr > other.ptr;}
	bool operator<=(const StorageIterator&other) const {return ptr <= other.ptr;}
	bool operator>=(const StorageIterator&other) const {return ptr >= other.ptr;}

	reference operator[](difference_type n) const {return *(ptr + n);}

	difference_type operator-(const StorageIterator&other) const {
		return ptr - other.ptr;
	}

private:
	pointer ptr;
};

#endif //STORAGE_HPP_
