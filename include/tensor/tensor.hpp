#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include"declarations.hpp"

#include"traits.hpp"
#include"storage.hpp"
#include"utils/tensor_slice.hpp"
#include"utils/tensor_utils.hpp"

template<typename T, std::size_t N>
class Tensor{
public:
	static constexpr std::size_t ord = N;
	using Value_type = T;

	Tensor() = default;
	Tensor(Tensor&&) = default;
	Tensor(const Tensor&) = default;
	Tensor&operator=(Tensor&&) = default;
	Tensor&operator=(const Tensor&) = default;
	~Tensor() = default;

	Tensor(TensorSlice<N> d, Storage<T> s) : desc_(d), elems_(s) {}

	template<typename U>
	Tensor(const Tensor<U,N>&x);

	template<typename U>
	Tensor&operator=(const Tensor<U,N>&x);

	template<typename... Exts>
	explicit Tensor(Exts... exts);

	Tensor(TensorInitializer<T,N>);
	Tensor& operator=(TensorInitializer<T,N>);

	template<typename U>
	Tensor(std::initializer_list<U>) = delete;
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
	x.elems_.share(this->elems_);
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

template<typename T, std::size_t N>
Tensor<T,N>::Tensor(TensorInitializer<T,N> init)
	: req_grad_{false}{
	desc_.start = 0;
	desc_.extents = tensor_impl::derive_extents<N>(init);
	desc_.size = tensor_impl::compute_strides(desc_.extents, desc_.strides);
	std::vector<T> temp;
	temp.reserve(desc_.size);
	tensor_impl::insert_flat(init, temp);

	std::cout << desc_.size << std::endl;
	std::cout << temp.size() << std::endl;
	assert(temp.size() == desc_.size);
	elems_ = storage::from_vector(temp);
}

template<typename T, std::size_t N>
Tensor<T,N>& Tensor<T,N>::operator=(TensorInitializer<T,N> init){
	req_grad_ = false;
	desc_.start = 0;
	desc_.extents = tensor_impl::derive_extents<N>(init);
	desc_.size = tensor_impl::compute_strides(desc_.extents, desc_.strides);
	std::vector<T> temp;
	temp.reserve(desc_.size);
	tensor_impl::insert_flat(init, temp);
	assert(temp.size() == desc_.size);
	elems_ = storage::from_vector(temp);
}
	



#endif //TENSOR_HPP_

