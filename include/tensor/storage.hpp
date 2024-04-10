#ifndef STORAGE_HPP_
#define STORAGE_HPP_

#include<cassert>
#include<algorithm>
#include<cstddef>
#include<ostream>
#include<iostream>
#include<iomanip>

template<typename T>
class StorageIterator;

//something between std::array and std::shared_ptr
template<typename T>
class Storage{
public:
	using value_type = T;

	using iterator = StorageIterator<T>;
	using const_iterator = StorageIterator<const T>;

	iterator begin() {return iterator(data_);}
	iterator end() {return iterator(data_ + size_);}

	const_iterator begin() const {return const_iterator(data_);}
	const_iterator end() const {return const_iterator(data_ + size_);}

	Storage(){
		this->size_ = 0;
		this->data_ = nullptr;
		this->observers_ = new unsigned int(1);
		//this->observers_ = nullptr;
	}

	explicit Storage(std::size_t n) 
		: size_(n), observers_(new unsigned int(1)){
		data_ = new T[n];
	}

	~Storage(){
		decrement_observers();
	}

	Storage(const Storage& other){
		this->deep_copy(other);
	}

	Storage&operator=(const Storage&other){
		if(this != &other){
			decrement_observers();
			this->deep_copy(other);
		}
		return*this;
	}

	Storage(Storage&&other) noexcept 
		: data_(other.data_), size_(other.size_), 
		observers_(other.observers_){
		other.data_ = nullptr;
		other.observers_ = nullptr;
		other.size_ = 0;
	}

	Storage&operator=(Storage&&other) noexcept{
		if(this != &other){
			decrement_observers();

			this->data_ = other.data_;
			this->size_ = other.size_;
			this->observers_ = other.observers_;

			other.data_ = nullptr;
			other.observers_ = nullptr;
			other.size_ = 0;
		}
		return*this;
	}

	Storage(std::initializer_list<T> list)
		: data_(new T[list.size()]), size_(list.size()),
	       	observers_(new unsigned int(1)){
		std::copy(list.begin(), list.end(), this->begin());
	}

	void share(Storage&sharee){
		if(this != &sharee){
			sharee.decrement_observers();
		}
		sharee.shared_copy(*this);
	}

	T&operator[](std::size_t i){
		return data_[i];
	}

	const T&operator[](std::size_t i) const{
		return data_[i];
	}

	std::size_t size(){
		return this->size_;
	}

	T*data(){
		return this->data_;
	}

	unsigned int observers(){
		return *this->observers_;
	}

private:
	T*data_;
	std::size_t size_;
	unsigned int*observers_;

	void decrement_observers(){
		--(*this->observers_);
		if(*this->observers_ == 0){
			if(data_){
				delete[] data_;
				data_ = nullptr;
			}
			delete observers_;
			observers_ = nullptr;
		}
	}

	void deep_copy(const Storage&other){
		this->size_ = other.size_;
		if(size_ > 0){
			this->data_ = new T[this->size_];
			std::copy(other.begin(), other.end(), this->begin());
		}
		else{
			data_ = nullptr;
		}
		this->observers_ = new unsigned int(1);
	}

	void shared_copy(const Storage&other){
		this->data_ = other.data_;
		this->size_ = other.size_;
		this->observers_ = other.observers_;
		++(*this->observers_);
		assert(this->data_ == other.data_);
	}
};

template<typename T>
bool same_storage(Storage<T>&a, Storage<T>&b){
	//assert(same type a, b)
	return a.data() == b.data();
}

template<typename T, typename U>
inline bool operator==(const Storage<T>&a, const Storage<U>&b){
	//assert T == U
	//if(a.size() == b.size()){
		auto ait = a.begin();
		for(auto bit = b.begin(); bit != b.end(); ++bit){
			if(*ait != *bit){
				return false;
			}
			++ait;
		}
	//}
	return true;
}

template<typename T, typename U>
inline bool operator!=(const Storage<T> a, const Storage<U>b){
	//assert T == U
	return !(a == b);
}

template<typename T>
std::ostream&operator<<(std::ostream&os, const Storage<T>s){
	for(auto it = s.begin(); it != s.end(); ++it){
		os << std::setfill('O') << *it << " ";
	}
	return os;
}

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

	StorageIterator&operator++() {ptr++; return*this;}
	StorageIterator operator++(int) {
		StorageIterator temp = *this; 
		++(*this);
		return temp;
	}
	StorageIterator&operator--() {ptr--; return*this;}
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
