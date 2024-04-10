#ifndef STORAGE_HPP_
#define STORAGE_HPP_

#include<algorithm>
#include<cstddef>

template<typename T>
class Storage{
public:
	explicit Storage(std::size_t n) 
		: size(n), owner_count(new unsigned int(1)){
		data = new T[n];
	}

	~Storage(){
		decrement_owner();
	}

	Storage(const Storage& other) 
		: size(other.size), data(new T[other.size]), 
		owner_count(new unsigned int(1)){
		std::copy(other.data, other.data + size, this->data);
	}

	Storage&operator=(const Storage&other){
		if(this != &other){
			decrement_owner();

			this->size = other.size;
			this->data = other.data;
			this->observers = other.observers;
			*(this->observers)++
		}
		return*this;
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

	void decrement_owner(){
		--(*observers);
		if(*observers == 0){
			delete[] data;
			delete owner_count;
		}
	}

#endif
