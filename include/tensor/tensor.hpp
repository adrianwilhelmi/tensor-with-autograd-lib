#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include"declarations.hpp"

#include"storage.hpp"
#include"traits.hpp"
#include"utils/tensor_slice.hpp"
#include"utils/tensor_utils.hpp"

template<typename T>
class Tensor{
public:
	Tensor() = default;
	Tensor(Tensor&&) = default;
	Tensor(const Tensor&) = default;
	Tensor&operator=(Tensor&&) = default;
	Tensor&operator=(const Tensor&&) = default;
	~Tensor() = default;

	Tensor(tensor_slice d, Storage<T> s) : desc_(d), elems_(s) {}

	template<typename U>
	Tensor(const Tensor<U>&x);

	template<typename U>
	Tensor&operator=(const Tensor<U>&x);

	template<typename... Exts>
	explicit Tensor(Exts... exts);

	template<typename U>
	Tensor(std::initializer_list<U>&list){
		std::vector<T> temp;
		initialize(list, temp, true);
	}	
	template<typename U>
	Tensor& operator=(std::initializer_list<U>) = delete;

	tensor_slice descriptor(){
		return this->desc_;
	}
private:
	template<typename U>
	void initialize(const std::initializer_list<U>&list, 
			std::vector<T>&temp, bool is_outer = false){
		for(const auto&elem : list){
			if constexpr(std::is_same<U,T>::value){
				temp.add(elem);
			}
			else{
				initialize(elem, temp, false);
			}
		}
		if(is_outer){
			//desc_ extents add (list.size())
		}
		if constexpr(std::is_same<U,T>::value){
			elems_ = storage::from_vector(temp);
		}
	}

private:
	tensor_slice desc_;
	Storage<T> elems_;
	Storage<T> grads_;

	bool req_grad_;
};

template<typename T>
template<typename U>
Tensor<T>::Tensor(const Tensor<U>&x)
	: req_grad_(false){
	static_assert(Convertible<U,T>(), "inconsistent types");
	this->desc_ = x.desc_;
	std::copy(x.begin(), x.end(), this->begin());
}

template<typename T>
template<typename U>
Tensor<T>& Tensor<T>::operator=(const Tensor<U>&x){
	static_assert(Convertible<U,T>(), "inconsistent types");
	this->req_grad_ = false;
	this->desc_ = x.desc_;
	this->elems_(desc_.size);
	this->elems_ = x.elems_;
	return*this;
}

template<typename T>
template<typename... Exts>
Tensor<T>::Tensor(Exts... exts)
	: desc_{exts...},
	elems_(desc_.size),
	grads_(),
	req_grad_(false)
{}
	


#endif //TENSOR_HPP_

