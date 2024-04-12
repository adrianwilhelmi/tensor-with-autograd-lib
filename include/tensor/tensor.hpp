#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include"declarations.hpp"

#include"storage.hpp"
#include"traits.hpp"
#include"utils/tensor_slice.hpp"
#include"utils/tensor_utils.hpp"

template<typename T, std::size_t N>
class Tensor{
public:
	Tensor() = default;
	Tensor(Tensor&&) = default;
	Tensor(const Tensor&) = default;
	Tensor&operator=(Tensor&&) = default;
	Tensor&operator=(const Tensor&&) = default;
	~Tensor() = default;

	Tensor(TensorSlice<N> d, Storage<T> s) : desc_(d), elems_(s) {}

	template<typename U>
	Tensor(const Tensor<U,N>&x);

	template<typename U>
	Tensor&operator=(const Tensor<U,N>&x);

	template<typename... Exts>
	explicit Tensor(Exts... exts);

	template<typename U>
	Tensor(std::initializer_list<U>&list){
		std::vector<T> temp;
		initialize(list, temp, true);
	}	
	template<typename U>
	Tensor& operator=(std::initializer_list<U>) = delete;

	TensorSlice<N> descriptor(){
		return this->desc_;
	}

private:
	TensorSlice<N> desc_;
	Storage<T> elems_;
	Storage<T> grads_;

	bool req_grad_;
};

template<typename T, std::size_t N>
template<typename U>
Tensor<T,N>::Tensor(const Tensor<U,N>&x)
	: req_grad_(false){
	static_assert(Convertible<U,T>(), "inconsistent types");
	this->desc_ = x.desc_;
	std::copy(x.begin(), x.end(), this->begin());
}

template<typename T, std::size_t N>
template<typename U>
Tensor<T,N>& Tensor<T,N>::operator=(const Tensor<U,N>&x){
	static_assert(Convertible<U,T>(), "inconsistent types");
	this->req_grad_ = false;
	this->desc_ = x.desc_;
	this->elems_(desc_.size);
	this->elems_ = x.elems_;
	return*this;
}

template<typename T, std::size_t N>
template<typename... Exts>
Tensor<T,N>::Tensor(Exts... exts)
	: desc_{exts...},
	elems_(desc_.size),
	grads_(),
	req_grad_(false)
{}
	


#endif //TENSOR_HPP_

