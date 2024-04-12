#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include"declarations.hpp"

#include"traits.hpp"
#include"storage.hpp"
#include"utils/tensor_slice.hpp"
#include"utils/tensor_utils.hpp"
#include"tensor_iterator.hpp"

template<typename T, std::size_t N>
class Tensor{
public:
	static constexpr std::size_t ord = N;
	using Value_type = T;

	using iterator = TensorIterator<T,N>;
	using const_iterator = TensorIterator<const T,N>;

	iterator begin() {return {this->desc_, elems_.data()};}
	iterator end() {return {this->desc_, elems_.data(), true};}
	const_iterator begin() const {return {this->desc_, elems_.data()};}
	const_iterator end() const {return {this->desc_, elems_.data(), true};}

	//constructors

	Tensor() = default;
	Tensor(Tensor&&) = default;
	Tensor(const Tensor&) = default;
	Tensor&operator=(Tensor&&) = default;
	Tensor&operator=(Tensor&x){
		this->req_grad_ = false;
		this->desc_ = x.desc_;
		x.elems_.share(this->elems_);
		return*this;
	}
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

	//indexing
	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(), T&>
	operator()(Args... args);
	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(), const T&>
	operator()(Args... args) const;

	TensorRef<T,N-1> dimslice(const std::size_t n, const std::size_t m);
	TensorRef<const T,N-1> dimslice(std::size_t n, std::size_t m) const;

	//misc
	Tensor<T,N>&sort_() {std::sort(this->begin(), this->end()); return*this;}
	Tensor<T,N> sort() {
		Tensor<T,N> copy(*this);
		std::sort(copy.begin(), copy.end());
		return copy;
	}
	
	T*data() {return elems_.data();}
	const T*data() const {return elems_.data();}

	bool empty() const {return this->begin() == this->end();}

	std::size_t order() const{return N;}

	std::size_t extent(const std::size_t i) const{
		return N >= i ? desc_.extents[i] : 0;
	}

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
	
//printing
template<typename M>
Enable_if<Tensor_type<M>(), std::ostream&>
operator<<(std::ostream& os, const M& m){
	if constexpr(M::ord == 1){
		os << "{";
		for(std::size_t i = 0; i != m.order(); ++i){
			os << std::setw(6) << m(i);
			if(i + 1 != m.order()) os << ",";
		}
		os << "}" << std::endl;
	}
	else{
		for(std::size_t i = 0; i != m.extent(0); ++i){
			os << "{";
			for(std::size_t j = 0; j != m.extent(1); ++j){
				os << std::setw(6) << std::fixed <<
				       std::setprecision(2) << m(i, j);
				if(j + 1 != m.extent(1)) os << ",";
			}
			os << "}";
			if(i + 1 != m.extent(0)) os << ',';
			os << std::endl;
		}
	}
	return os;
}

template<typename T, std::size_t N>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), T&>
Tensor<T,N>::operator()(Args... args){
	assert(tensor_impl::check_bounds(this->desc_, args...));
	return*(data() + this->desc_(args...));
}	

template<typename T, std::size_t N>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), const T&>
Tensor<T,N>::operator()(Args... args) const{
	assert(tensor_impl::check_bounds(this->desc_, args...));
	return*(data() + this->desc_(args...));
}	


#endif //TENSOR_HPP_

