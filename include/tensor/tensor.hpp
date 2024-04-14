#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include"declarations.hpp"
#include"traits.hpp"
#include"storage.hpp"
#include"utils/tensor_slice.hpp"
#include"utils/tensor_utils.hpp"
#include"tensor_iterator.hpp"
#include"function.hpp"

#include<iostream>
#include<sstream>

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
		if(this != &x){
			this->req_grad_ = x.req_grad_;
			this->grad_fn_ = FunctionEmpty{};
			this->desc_ = x.desc_;
			x.elems_.share(this->elems_);
			if(x.req_grad_){
				x.grads_.share(this->grads_);
			}
		}
		return*this;
	}
	~Tensor() = default;

	Tensor(const TensorSlice<N>&d, Storage<T>&e, Storage<T>&g) 
		: desc_(d), grad_fn_(FunctionEmpty{}),req_grad_(true){
		e.share(this->elems_);
		g.share(this->grads_);
	}

	Tensor(const TensorSlice<N>& d, Storage<T>&e) : desc_(d){
		this->grad_fn_ = FunctionEmpty{};
		this->req_grad_ = false;
		e.share(this->elems_);
	}

	Tensor(const TensorSlice<N>& d) 
		: desc_(d), elems_(d.size), 
		grad_fn_(FunctionEmpty{}), req_grad_(false) {}

	template<typename U>
	Tensor(Tensor<U,N>&x);

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

	template<typename M>
	Enable_if<Tensor_type<M>(), void> share(M&m){
		m.req_grad_ = this->req_grad_;
		m.desc_ = this->desc_;
		this->elems_.share(m.elems_);
		if(m.req_grad_){
			this->grads_.share(m.grads_);
		}
	}

	//indexing
	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(), T&>
	operator()(Args... args);
	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(), const T&>
	operator()(Args... args) const;

	Tensor<T,N-1> dimslice(const std::size_t n, const std::size_t m);
	Tensor<const T,N-1> dimslice(const std::size_t n, const std::size_t m) const;

	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<T,N>>
	dimslices(std::size_t dim, Args... args);
	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<const T,N>>
	dimslices(std::size_t dim, Args... args) const;

	Tensor<T,N> dimslices_arange(std::size_t dim, 
			std::size_t from, std::size_t to);
	Tensor<const T,N> dimslices_arange(std::size_t dim, 
			std::size_t from, std::size_t to) const;

	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(),
		Tensor<T,sizeof...(Args)>>
	view(Args... args);
	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(),
		Tensor<const T,sizeof...(Args)>>
	view(Args... args) const;

	Tensor<T,N-1> operator[](std::size_t i) {return dimslice(0, i);}
	Tensor<const T,N-1> operator[](std::size_t i) const {return dimslice(0, i);}

	Tensor<T,N-1> row(const std::size_t i) {return dimslice(0, i);}
	Tensor<const T,N-1> row(const std::size_t i) const {return dimslice(0, i);}

	Tensor<T,N-1> col(const std::size_t i) {return dimslice(1, i);}
	Tensor<const T,N-1> col(const std::size_t i) const {return dimslice(1, i);}

	template<std::size_t NN = N, typename = Enable_if<(NN == 1)>>
	T& row(std::size_t i) {return elems_[i];}

	template<std::size_t NN = N, typename = Enable_if<(NN == 1)>>
	T& col(std::size_t i) = delete;

	//misc
	T sum() const {return std::accumulate(this->begin(), this->end(), T{0});}
	T mean() const {return this->sum() / this->size();}
	T max() const {return*std::max_element(this->begin(), this->end());}
	T min() const {return*std::min_element(this->begin(), this->end());}
	T median() const{
		Storage<T> sorted(this->elems_);
		std::sort(sorted.begin(), sorted.end());
		std::size_t mid = sorted.size() / 2;
		if(sorted.size() % 2 == 0){
			return(sorted[mid] + sorted[mid - 1]) / T(2);
		}
		else{
			return sorted[mid];
		}
	}

	bool empty() const {return begin() == end();}

	Tensor<T,N>&sort_() {std::sort(this->begin(), this->end()); return*this;}
	Tensor<T,N> sort() {
		Tensor<T,N> copy(*this);
		std::sort(copy.begin(), copy.end());
		return copy;
	}
	
	T*data() {return elems_.data();}
	const T*data() const {return elems_.data();}

	std::size_t order() const{return N;}
	std::size_t extent(const std::size_t i) const{return N >= i ? desc_.extents[i] : 0;}
	std::size_t size() const{return this->desc_.size;}

	TensorSlice<N> descriptor() const{
		return this->desc_;
	}

	const Storage<T>&storage() const{
		return this->elems_;
	}
	
	Storage<T>&storage(){
		return this->elems_;
	}

	template<typename U>
	Tensor<U,N> convert() const{
		Storage<U> s = this->elems_.convert();
		TensorSlice<N> d = this->desc_;
		if(this->req_grad_){
			Storage<U> g = this->grads_.convert();
			return {d, s, g};
		}
		return {d, s};
	}
	
	//autograd
	bool requires_grad() const{
		return this->req_grad_;
	}

	void init_grad(){
		this->grads_.reset(elems_.size());
		this->grads_.fill(T(0));
	}

private:
	TensorSlice<N> desc_;
	Storage<T> elems_;
	Storage<T> grads_;
	func_variant<T,N> grad_fn_;
	bool req_grad_;
};

template<typename T>
class Tensor<T,0>{
public:
	std::size_t order() const{return 0;}
	std::size_t extent(const std::size_t i) const{return 0 == i ? 1 : 0;}

private:
	T*elem;
};


template<typename T, std::size_t N>
template<typename U>
Tensor<T,N>::Tensor(Tensor<U,N>&x)
	: grad_fn_(FunctionEmpty{}),
	req_grad_(false){
	static_assert(Convertible<U,T>(), "inconsistent types");
	this->desc_ = x.descriptor();
	std::copy(x.begin(), x.end(), this->begin());
}

template<typename T, std::size_t N>
template<typename U>
Tensor<T,N>& Tensor<T,N>::operator=(const Tensor<U,N>&x){
	static_assert(Convertible<U,T>(), "inconsistent types");
	this->req_grad_ = false;
	this->grad_fn_ = FunctionEmpty{};
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
  	grad_fn_(FunctionEmpty{}),
	req_grad_(false)
{}

template<typename T, std::size_t N>
Tensor<T,N>::Tensor(TensorInitializer<T,N> init)
	: req_grad_{false}{
	grad_fn_ = FunctionEmpty{};
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
	grad_fn_ = FunctionEmpty{};
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
template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const Tensor<T, N>& tensor) {
	auto desc = tensor.descriptor();
	auto it = tensor.begin();
	if(tensor.order() == 1){
		os << "{";
		for(std::size_t i = 0; i < desc.extents[0] - 1; ++i){
			os << *it << ", ";
			++it;
		}
		os << *it << "}" << std::endl;
	}
	else if(tensor.order() == 2){
		for(std::size_t i = 0; i < desc.extents[0]; ++i){
			os << "{";
			for(std::size_t j = 0; j < desc.extents[1] - 1; ++j){
				os << *it << ", ";
				++it;
			}
			os << *it << "}";
			++it;
			os << std::endl;
		}
	}
	else if(tensor.order() == 3){
		for(std::size_t i = 0; i < desc.extents[0]; ++i){
			os << "{" << std::endl;
			for(std::size_t j = 0; j < desc.extents[1]; ++j){
				os << "\t{";
				for(std::size_t k = 0; k < desc.extents[2] - 1; ++k){
					os << *it << ", ";
					++it;
				}
				os << *it << "},";
				++it;
				os << std::endl;
			}
			os << "}" << std::endl;
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

template<typename T, std::size_t N>
Tensor<T,N-1> Tensor<T,N>::dimslice(const std::size_t n, const std::size_t m){
	assert(n < N);
	assert(m < extent(n));
	TensorSlice<N-1> ts;

	int j = 0;
	for(std::size_t i = 0; i < n; ++i){
		ts.extents[j] = this->desc_.extents[i];
		ts.strides[j] = this->desc_.strides[i];
		++j;
	}

	for(std::size_t i = n + 1; i < N; ++i){
		ts.extents[j] = this->desc_.extents[i];
		ts.strides[j] = this->desc_.strides[i];
		++j;
	}

	ts.start = this->desc_.start + m * this->desc_.strides[n];
	ts.size = tensor_impl::compute_size(ts.extents);

	if(this->req_grad_)
		return{ts, this->elems_, this->grads_};
	return{ts, this->elems_};
}

template<typename T, std::size_t N>
Tensor<const T,N-1> Tensor<T,N>::dimslice(const std::size_t n, 
		const std::size_t m) const{
	assert(n < N);
	assert(m < extent(n));
	TensorSlice<N-1> ts;

	int j = 0;
	for(std::size_t i = 0; i < n; ++i){
		ts.extents[j] = this->desc_.extents[i];
		ts.strides[j] = this->desc_.strides[i];
		++j;
	}

	for(std::size_t i = n + 1; i < N; ++i){
		ts.extents[j] = this->desc_.extents[i];
		ts.strides[j] = this->desc_.strides[i];
		++j;
	}

	ts.start = this->desc_.start + m * this->desc_.strides[n];
	ts.size = tensor_impl::compute_size(ts.extents);
	

	if(this->req_grad_)
		return{ts, this->elems_, this->grads_};
	return{ts, this->elems_};
	/*
	if(this->req_grad_){
		return Tensor<T,N-1>(ts, this->elems_, this->grads_);
	}
	return Tensor<const T,N-1>(ts, this->elems_);
	*/
}

template<typename T, std::size_t N>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<T,N>>
Tensor<T,N>::dimslices(std::size_t dim, Args... args){
	assert(dim < order());
	assert(sizeof...(args) <= extent(dim));
	assert(sizeof...(args) < 3); //FOR NOW WORKS ONLY WITH 2 SUBTENSORS

	TensorSlice<N> d;

	d.extents = this->desc_.extents;
	d.extents[dim] = sizeof...(args);
	d.strides = this->desc_.strides;
	d.size = tensor_impl::compute_size(d.extents);

	std::array<std::size_t, sizeof...(args)> indexes = {
		static_cast<std::size_t>(args)...};

	d.start = this->desc_.start;
	std::sort(indexes.begin(), indexes.end());
	d.start += indexes.front() * this->desc_.strides[dim];

	d.strides[dim] *= (indexes[1] - indexes[0]);

	if(this->req_grad_)
		return{d, this->elems_, this->grads_};
	return{d, this->elems_};
}

template<typename T, std::size_t N>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<const T,N>>
Tensor<T,N>::dimslices(std::size_t dim, Args... args) const{
	assert(dim < order());
	assert(sizeof...(args) <= extent(dim));
	assert(sizeof...(args) < 3); //FOR NOW WORKS ONLY WITH 2 SUBTENSORS

	TensorSlice<N> d;

	d.extents = this->desc_.extents;
	d.extents[dim] = sizeof...(args);
	d.strides = this->desc_.strides;
	d.size = tensor_impl::compute_size(d.extents);

	std::array<std::size_t, sizeof...(args)> indexes = {
		static_cast<std::size_t>(args)...};

	d.start = this->desc_.start;
	std::sort(indexes.begin(), indexes.end());
	d.start += indexes.front() * this->desc_.strides[dim];
	
	d.strides[dim] *= (indexes[1] - indexes[0]);

	if(this->req_grad_)
		return{d, this->elems_, this->grads_};
	return{d, this->elems_};
}

template<typename T, std::size_t N>
Tensor<T,N> Tensor<T,N>::dimslices_arange(std::size_t dim,
		std::size_t from, std::size_t to){
	assert(dim < order());
	assert(from <= to);
	assert(to <= this->extent(dim));

	TensorSlice<N> d;

	d.extents = this->desc_.extents;
	d.extents[dim] = to - from + 1;
	d.strides = this->desc_.strides;
	d.size = tensor_impl::compute_size(d.extents);

	d.start = this->desc_.start + from * this->desc.strides[dim];

	if(this->req_grad_)
		return{d, this->elems_, this->grads_};
	return{d, this->elems_};
}

template<typename T, std::size_t N>
Tensor<const T,N> Tensor<T,N>::dimslices_arange(std::size_t dim,
		std::size_t from, std::size_t to) const{
	assert(dim < order());
	assert(from <= to);
	assert(to <= this->extent(dim));

	TensorSlice<N> d;

	d.extents = this->desc_.extents;
	d.extents[dim] = to - from + 1;
	d.strides = this->desc_.strides;
	d.size = tensor_impl::compute_size(d.extents);

	d.start = this->desc_.start + from * this->desc.strides[dim];

	if(this->req_grad_)
		return{d, this->elems_, this->grads_};
	return{d, this->elems_};
}

template<typename T, std::size_t N>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<T,sizeof...(Args)>>
Tensor<T,N>::view(Args... args){
	std::size_t args_product = (... * args);
	std::size_t exts_product = std::accumulate(this->desc_.extents.begin(),
			this->desc_.extents.end(), 1, [](std::size_t a,
				std::size_t b) {return a * b;});

	assert(args_product == exts_product);

	std::array<std::size_t, sizeof...(Args)> exts{
		static_cast<std::size_t>(args)...};
	TensorSlice<sizeof...(Args)> d{exts};

	if(this->req_grad_)
		return{d, this->elems_, this->grads_};
	return{d, this->elems_};
}

template<typename T, std::size_t N>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<const T,sizeof...(Args)>>
Tensor<T,N>::view(Args... args) const{
	std::size_t args_product = (... * args);
	std::size_t exts_product = std::accumulate(this->desc_.extents.begin(),
			this->desc_.extents.end(), 1, [](std::size_t a,
				std::size_t b) {return a * b;});

	assert(args_product == exts_product);

	std::array<std::size_t, sizeof...(Args)> exts{
		static_cast<std::size_t>(args)...};
	TensorSlice<sizeof...(Args)> d{exts};

	if(this->req_grad_)
		return{d, this->elems_, this->grads_};
	return{d, this->elems_};
}


#endif //TENSOR_HPP_

