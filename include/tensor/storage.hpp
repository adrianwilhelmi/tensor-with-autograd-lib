#ifndef STORAGE_HPP_
#define STORAGE_HPP_

#include<algorithm>
#include<cstddef>

template<typename T>
class StorageIterator;

//something between std::array and std::shared_ptr
template<typename T>
class Storage{
public:
	using value_type = T;

	using iterator = StorageIterator<T>;
	using const_iterator = StorageIterator<const T>;

	iterator begin() {return iterator(data);}
	iterator end() {return iterator(data + size);}

	const_iterator begin() const {return const_iterator(data);}
	const_iterator end() const {return const_iterator(data + size);}

	explicit Storage(std::size_t n) 
		: size(n), observers(new unsigned int(1)){
		data = new T[n];
	}

	~Storage(){
		decrement_observers();
	}

	Storage(const Storage& other) 
		: size(other.size), data(new T[other.size]), 
		observers(new unsigned int(1)){
		std::copy(other.data, other.data + size, this->data);
	}

	Storage&operator=(const Storage&other){
		if(this != &other){
			decrement_observers();

			this->size = other.size;
			this->data = other.data;
			this->observers = other.observers;
			++(*this->observers);
		}
		return*this;
	}

	Storage(std::initializer_list<T> list)
		: data(new T[list.size()]), size(list.size()),
	       	observers(new unsigned int(1)){
		std::copy(list.begin(), list.end(), data);
	}

	T&operator[](std::size_t i){
		return data[i];
	}

	const T&operator[](std::size_t i) const{
		return data[i];
	}

	std::size_t size(){
		return this->size;
	}

	T*data(){
		return this->data;
	}

private:
	T*data;
	std::size_t size;
	unsigned int*observers;

	void decrement_observers(){
		--(*observers);
		if(*observers == 0){
			delete[] data;
			delete observers;
		}
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
	StorageIterator&operator--() {ptr++; return*this;}
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

	bool operator==(const StorageIterator&other) const {return ptr == other.ptr}
	bool operator!=(const StorageIterator&other) const {return ptr != other.ptr}
	bool operator<(const StorageIterator&other) const {return ptr < other.ptr}
	bool operator>(const StorageIterator&other) const {return ptr > other.ptr}
	bool operator<=(const StorageIterator&other) const {return ptr <= other.ptr}
	bool operator>=(const StorageIterator&other) const {return ptr >= other.ptr}

	reference operator[](difference_type n) const {return *(ptr + n);}

	difference_type operator-(const StorageIterator&other) const {
		return ptr - other.ptr;
	}

private:
	pointer ptr;
};

#endif
