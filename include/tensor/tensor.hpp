#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include"declarations.hpp"
#include"traits.hpp"
#include"storage.hpp"
#include"utils/tensor_slice.hpp"
#include"utils/tensor_utils.hpp"
#include"tensor_iterator.hpp"
#include"function.hpp"
#include"node.hpp"

#include<cassert>
#include<iostream>
#include<sstream>

template<typename T>
class Tensor{
public:
	using Value_type = T;

	using iterator = TensorIterator<T>;
	using const_iterator = TensorIterator<const T>;

	iterator begin() {return {this->desc_, elems_.data()};}
	iterator end() {return {this->desc_, elems_.data(), true};}
	const_iterator begin() const {return {this->desc_, elems_.data()};}
	const_iterator end() const {return {this->desc_, elems_.data(), true};}

	//constructors

	Tensor() = default;
	Tensor(Tensor&&) = default;
	Tensor(const Tensor& x){
		this->req_grad_ = false;
		this->desc_ = x.desc_;
		this->elems_ = x.elems_;

		this->node = nullptr;
	}

	Tensor&operator=(Tensor&&) = default;
	Tensor&operator=(Tensor&x){
		if(this != &x){
			this->req_grad_ = x.req_grad_;
			this->desc_ = x.desc_;
			x.elems_.share(this->elems_);

			if(x.requires_grad()){
				this->node = std::make_shared<Node<T>>(*this);
				this->node->grad_fn = FunctionId<T>{};
				this->node->set_inputs(x);
			}
		}
		return*this;
	}

	Tensor(const TensorSlice& d, Storage<T>&e) : desc_(d), req_grad_(false){
		e.share(this->elems_);
		this->node = nullptr;
	}

	Tensor(const TensorSlice& d)
		: desc_(d), elems_(d.size), 
		req_grad_(false) {
		this->node = nullptr;
	}

	Tensor<T> copy_dims(){
		Storage<T> elems(this->elems_.size());
		TensorSlice d = this->desc_;
		return{d, elems};
	}

	template<typename U>
	Tensor(const Tensor<U>&x);

	template<typename U>
	Tensor&operator=(const Tensor<U>&x);

	template<typename... Exts>
	explicit Tensor(Exts... exts);

	template<typename U>
	Tensor(std::initializer_list<U> init) = delete;
	template<typename U>
	Tensor& operator=(std::initializer_list<U> init) = delete;

	template<typename M>
	Enable_if<Tensor_type<M>(), void> share(M&m){
		m.req_grad_ = this->req_grad_;
		m.desc_ = this->desc_;
		this->elems_.share(m.elems_);
	}

	void fill(const T val){
		for(auto it = this->begin(); it != this->end(); ++it){
			*it = val;
		}
		/*
		for(auto& elem : *this){
			elem = val;
		}
		*/
	}

	//indexing
	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(), T&>
	operator()(Args... args);
	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(), const T&>
	operator()(Args... args) const;

	Tensor<T> operator()(TensorSlice d){
		return {d, this->elems_};
	}

	Tensor<T> dimslice(const std::size_t n, const std::size_t m);
	Tensor<const T> dimslice(const std::size_t n, const std::size_t m) const;

	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<T>>
	dimslices(std::size_t dim, Args... args);
	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<const T>>
	dimslices(std::size_t dim, Args... args) const;

	Tensor<T> dimslices_arange(std::size_t dim, 
			std::size_t from, std::size_t to);
	Tensor<const T> dimslices_arange(std::size_t dim, 
			std::size_t from, std::size_t to) const;

	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(),
		Tensor<T>>
	view(Args... args);
	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(),
		Tensor<const T>>
	view(Args... args) const;

	Tensor<T> operator[](std::size_t i) {return dimslice(0, i);}
	Tensor<const T> operator[](std::size_t i) const {return dimslice(0, i);}

	Tensor<T> row(const std::size_t i) {return dimslice(0, i);}
	Tensor<const T> row(const std::size_t i) const {return dimslice(0, i);}

	Tensor<T> col(const std::size_t i) {return dimslice(1, i);}
	Tensor<const T> col(const std::size_t i) const {return dimslice(1, i);}

	//tensor ops
	//tensor scalar ops
	
	template<typename F>
	Tensor& apply(F f);

	template<typename M, typename F>
	Enable_if<Tensor_type<M>(), Tensor&> apply(const M&m, F f);

	Tensor& operator=(const T& value);

	Tensor& operator+=(const T& value);
	Tensor& operator-=(const T& value);
	Tensor& operator*=(const T& value);
	Tensor& operator/=(const T& value);
	Tensor& operator%=(const T& value);

	//element-wise ops
	template<typename M>
	Enable_if<Tensor_type<M>(), Tensor&> operator+=(const M&x);
	template<typename M>
	Enable_if<Tensor_type<M>(), Tensor&> operator-=(const M&x);
	template<typename M>
	Enable_if<Tensor_type<M>(), Tensor&> operator*=(const M&x);
	template<typename M>
	Enable_if<Tensor_type<M>(), Tensor&> operator/=(const M&x);
	template<typename M>
	Enable_if<Tensor_type<M>(), Tensor&> operator%=(const M&x);

	template<typename U = typename std::remove_const<T>::type>
	Tensor<U> operator-();

	//2d tensor
	Tensor<T> diag(){
		assert(this->order() == 2 && this->extent(0) == this->extent(1));
		
		TensorSlice diag_slice;
		diag_slice.size = desc_.extents[0];
		diag_slice.start = desc_.start;
		diag_slice.extents[0] = desc_.extents[0];
		diag_slice.strides[0] = desc_.strides[0] + desc_.strides[1];

		Tensor<T> res(diag_slice, this->elems);

		if(this->req_grad_){
			res.enable_grad();
			func_variant<T> fn = FunctionId<T>{};

			auto n = std::make_shared<Node<T>>(res);
			n->grad_fn = fn;
			n->set_inputs(*this);

			res.set_node(n);
		}
		return res;
	}

	Tensor<const T> diag() const{
		assert(this->order() == 2 && this->extent(0) == this->extent(1));
		
		TensorSlice diag_slice;
		diag_slice.size = desc_.extents[0];
		diag_slice.start = desc_.start;
		diag_slice.extents[0] = desc_.extents[0];
		diag_slice.strides[0] = desc_.strides[0] + desc_.strides[1];

		Tensor<T> res(diag_slice, this->elems_);

		if(this->req_grad_){
			res.enable_grad();
			func_variant<T> fn = FunctionId<T>{};

			auto n = std::make_shared<Node<T>>(res);
			n->grad_fn = fn;
			n->set_inputs(*this);

			res.set_node(n);
		}
		return res;
	}

	Tensor<T>& transpose_(std::size_t d1, std::size_t d2){
		assert(d1 < this->order() && d2 < this->order());
		std::swap(this->desc.extents[d1], this->desc.extents[d2]);
		std::swap(this->desc.strides[d1], this->desc.strides[d2]);
		return*this;
	}

	Tensor<T>& transpose_(){
		return transpose_(0,1);
	}

	Tensor<T> transpose(std::size_t d1, std::size_t d2){
		assert(d1 < this->order() && d2 < this->order());

		TensorSlice d;
		d.start = this->desc_.start;
		d.extents = this->desc_.extents;
		d.strides = this->desc_.strides;

		std::swap(d.extents[d1], d.extents[d2]);
		std::swap(d.strides[d1], d.strides[d2]);

		Tensor<T> res(d, this->elems_);
		if(this->req_grad_){
			res.enable_grad();
			func_variant<T> fn = FunctionId<T>{};

			auto n = std::make_shared<Node<T>>(res);
			n->grad_fn = fn;
			n->set_inputs(*this);

			res.set_node(n);
		}
		return res;
	}

	Tensor<T> transpose(){
		return transpose(0,1);
	}

	Tensor<const T> transpose(std::size_t d1, std::size_t d2) const{
		assert(d1 < this->order() && d2 < this->order());

		TensorSlice d;
		d.start = this->desc_.start;
		d.extents = this->desc_.extents;
		d.strides = this->desc_.strides;

		std::swap(d.extents[d1], d.extents[d2]);
		std::swap(d.strides[d1], d.strides[d2]);

		Tensor<T> res(d, this->elems_);
		if(this->req_grad_){
			res.enable_grad();
			func_variant<T> fn = FunctionId<T>{};

			auto n = std::make_shared<Node<T>>(res);
			n->grad_fn = fn;
			n->set_inputs(*this);

			res.set_node(n);
		}
		return res;
	}

	Tensor<const T> transpose() const{
		return transpose(0,1);
	}


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

	Tensor<T>&sort_() {std::sort(this->begin(), this->end()); return*this;}
	Tensor<T> sort() {
		Tensor<T> copy(*this);
		std::sort(copy.begin(), copy.end());
		return copy;
	}
	
	T*data() {return elems_.data();}
	const T*data() const {return elems_.data();}

	std::size_t order() const{return this->desc_.extents.size();}
	std::size_t extent(const std::size_t i) const{return order() >= i ? desc_.extents[i] : 0;}
	std::size_t size() const{return this->desc_.size;}

	TensorSlice descriptor() const{
		return this->desc_;
	}

	const Storage<T>&storage() const{
		return this->elems_;
	}
	
	Storage<T>&storage(){
		return this->elems_;
	}

	template<typename U>
	Tensor<U> convert() const{
		Storage<U> s = this->elems_.convert();
		TensorSlice d = this->desc_;
		Tensor<T> res(d, s);
		if(this->req_grad_){
			res.enable_grad();
			auto n = std::make_shared<Node<T>>(res);
			func_variant<T> fn = FunctionId<T>{};
			n->grad_fn = fn;
			res.set_node(n);
		}
		return res;
	}

	//funcs
	Tensor<T> pow(Tensor<T>& exps){
		assert(same_extents(this->desc_, exps.descriptor()) 
				|| exps.order() == 0);

		Tensor<T> res(*this);
		for(auto i = res.begin(), j = exps.begin(); i != res.end(); ++i, ++j)
			*i = std::pow(*i, *j);

		if(this->req_grad_){
			res.enable_grad();
			func_variant<T> fn = FunctionPow<T>{};
			auto n = std::make_shared<Node<T>>(res);
			n->grad_fn = fn;
			n->set_inputs(*this, exps);

			res.set_node(n);
		}

		return res;
	}

	Tensor<T> pow(const T exp) const{
		Tensor<T> res(*this);
		res.apply([&](T& a) {a = std::pow(a, exp);});
		return res;
	}

	Tensor<T> log(){
		Tensor<T> res(*this);
		res.apply([&](T& a) {a = std::log(a);});

		if(this->req_grad_){
			res.enable_grad();
			func_variant<T> fn = FunctionLog<T>{};
			auto n = std::make_shared<Node<T>>(res);
			n->grad_fn = fn;
			n->set_inputs(*this);

			res.set_node(n);
		}
		
		return res;
	}
	
	Tensor<T> exp(){
		Tensor<T> res(*this);
		res.apply([&](T& a) {a = std::exp(a);});

		if(this->req_grad_){
			res.enable_grad();
			func_variant<T> fn = FunctionExp<T>{};
			auto n = std::make_shared<Node<T>>(res);
			n->grad_fn = fn;
			n->set_inputs(*this);

			res.set_node(n);
		}

		return res;
	}

	Tensor<T> relu() const{
		Tensor<T> res(*this);
		res.apply([&](T& a) {a = tensor_impl::relu<T>(a);});

		if(this->req_grad_){
			res.enable_grad();
			func_variant<T> fn = FunctionRelu<T>{};
			auto n = std::make_shared<Node<T>>(res);
			n->grad_fn = fn;
			n->set_inputs(*this);

			res.set_node(n);
		}

		return res;
	}

	Tensor<T> tanh() {
		Tensor<T> res(*this);
		res.apply([&](T& a) {a = tensor_impl::tanh<T>(a);});

		if(this->req_grad_){
			res.enable_grad();
			func_variant<T> fn = FunctionTanh<T>{};
			auto n = std::make_shared<Node<T>>(res);
			n->grad_fn = fn;
			n->set_inputs(*this);

			res.set_node(n);
		}

		return res;
	}

	Tensor<T> sigmoid() {
		Tensor<T> res(*this);
		res.apply([&](T& a) {a = tensor_impl::sigmoid<T>(a);});

		if(this->req_grad_){
			res.enable_grad();
			func_variant<T> fn = FunctionSigmoid<T>{};
			auto n = std::make_shared<Node<T>>(res);
			n->grad_fn = fn;
			n->set_inputs(*this);

			res.set_node(n);
		}

		return res;
	}

	Tensor<T> softmax() const{
		Tensor<T> res(*this);
		T max = res.max();
		res.apply([&](T& a) {a = std::exp(a - max);});
		T sum = res.sum();
		res.apply([&](T&a) {a /= sum;});

		return res;
	}

	//in-place
	Tensor<T>& pow_(Tensor<T>& exps){
		for(auto i = begin(), j = exps.begin(); i != end(); ++i, ++j)
			*i = std::pow(*i, *j);
		return*this;
	}
	Tensor<T>& pow_(T exp){
		return apply([&](T&a) {a = std::pow(a, exp);});
	}
	Tensor<T>& log_(T exp){
		return apply([&](T&a) {a = std::log(a);});
	}
	Tensor<T>& exp_(){
		return apply([&](T&a) {a = std::exp(a);});
	}
	Tensor<T>& relu_(){
		return apply([&](T& a) {a = tensor_impl::relu<T>(a);});
	}
	Tensor<T>& tanh_(){
		return apply([&](T& a) {a = tensor_impl::tanh<T>(a);});
	}
	Tensor<T>& sigmoid_(){
		return apply([&](T& a) {a = tensor_impl::sigmoid<T>(a);});
	}
	Tensor<T>& softmax_(){
		T max = this->max();
		this->apply([&](T&a) {a = std::exp(a - max);});
		T sum = this->sum();
		return apply([&](T&a) {a /= sum;});
	}

	
	//autograd
	bool requires_grad() const{
		return this->req_grad_;
	}

	void enable_grad(){
		this->req_grad_ = true;
		if(this->node == nullptr){
			this->node = std::make_shared<Node<T>>(*this);
		}
	}

	Tensor<T>& grad(){
		if(!req_grad_ || !this->node){
			throw std::runtime_error("grad is disabled");
		}
		return this->node->grads;
	}

	const Tensor<T>& grad() const{
		if(!req_grad_ || !this->node){
			throw std::runtime_error("grad is off");
		}
		return this->node->grads;
	}

	Tensor<T>& grad(TensorSlice d){
		if(!req_grad_ || !this->node){
			throw std::runtime_error("grad is off");
		}
		return this->node->grads(d);
	}

	const Tensor<T>& grad(TensorSlice d) const{
		if(!req_grad_ || !this->node){
			throw std::runtime_error("grad is off");
		}
		return this->node->grads(d);
	}

	void backward_(){
		this->node->backward();
	}

	void backward(){
		if(this->node->grads.size() == 0){
			this->node->init_grad();
		}
		this->node->grads.fill(T{1});
		this->node->backward();
	}

	void set_node(const std::shared_ptr<Node<T>>& n){
		this->node = n;
	}

	std::shared_ptr<Node<T>>& get_node(){
		return this->node;
	}
	const std::shared_ptr<Node<T>>& get_node() const{
		return this->node;
	}

private:
	TensorSlice desc_;
	Storage<T> elems_;
	std::shared_ptr<Node<T>> node;
	bool req_grad_;
};

template<typename T>
template<typename U>
Tensor<T>::Tensor(const Tensor<U>&x)
	: elems_(x.size()), req_grad_(false) {
	static_assert(Convertible<U,T>(), "inconsistent types");
	this->desc_ = x.descriptor();
	std::copy(x.begin(), x.end(), this->begin());

	this->node = nullptr;
}

template<typename T>
template<typename U>
Tensor<T>& Tensor<T>::operator=(const Tensor<U>&x){
	static_assert(Convertible<U,T>(), "inconsistent types");
	this->desc_ = x.desc_;
	x.elems_.share(this->elems_);

	if(x.req_grad_){
		this->req_grad_ = true;
		this->node = std::make_shared<Node<T>>(*this);
		this->node->grad_fn = FunctionId<T>{};
		this->node.set_inputs(x);
	}
	else{
		this->req_grad_ = false;
		this->node = nullptr;
	}

	return*this;
}

template<typename T>
template<typename... Exts>
Tensor<T>::Tensor(Exts... exts)
	: desc_{exts...},
	elems_(desc_.size),
	req_grad_(false){

	this->node = nullptr;
}

//printing
template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
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

template<typename T>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), T&>
Tensor<T>::operator()(Args... args){
	assert(tensor_impl::check_bounds(this->desc_, args...) && "index oob");
	return*(data() + this->desc_(args...));
}	

template<typename T>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), const T&>
Tensor<T>::operator()(Args... args) const{
	assert(tensor_impl::check_bounds(this->desc_, args...) && "index oob");
	return*(data() + this->desc_(args...));
}	

template<typename T>
Tensor<T> Tensor<T>::dimslice(const std::size_t n, const std::size_t m){
	assert(n < order());
	assert(m < extent(n));

	TensorSlice ts(order() - 1);

	int j = 0;
	for(std::size_t i = 0; i < n; ++i){
		ts.extents[j] = this->desc_.extents[i];
		ts.strides[j] = this->desc_.strides[i];
		++j;
	}

	for(std::size_t i = n + 1; i < order(); ++i){
		ts.extents[j] = this->desc_.extents[i];
		ts.strides[j] = this->desc_.strides[i];
		++j;
	}

	ts.start = this->desc_.start + m * this->desc_.strides[n];
	ts.compute_size();

	Tensor<T> res(ts, this->elems_);
	if(this->req_grad_){
		res.enable_grad();
		
		auto n = std::make_shared<Node<T>>(res);
		func_variant<T> fn = FunctionId<T>{};
		n->grad_fn = fn;
		n->set_inputs(*this);

		res.set_node(n);
	}
	return res;
}

template<typename T>
Tensor<const T> Tensor<T>::dimslice(const std::size_t n, 
		const std::size_t m) const{
	assert(n < order());
	assert(m < extent(n));
	TensorSlice ts(order() - 1);

	int j = 0;
	for(std::size_t i = 0; i < n; ++i){
		ts.extents[j] = this->desc_.extents[i];
		ts.strides[j] = this->desc_.strides[i];
		++j;
	}

	for(std::size_t i = n + 1; i < order(); ++i){
		ts.extents[j] = this->desc_.extents[i];
		ts.strides[j] = this->desc_.strides[i];
		++j;
	}

	ts.start = this->desc_.start + m * this->desc_.strides[n];
	ts.compute_size();

	Tensor<T> res(ts, this->elems_);
	if(this->req_grad_){
		res.enable_grad();
		func_variant<T> fn = FunctionId<T>{};

		auto n = std::make_shared<Node<T>>(res);
		n->grad_fn = fn;
		n->set_inputs(*this);

		res.set_node(n);
	}
	return res;
}

template<typename T>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<T>>
Tensor<T>::dimslices(std::size_t dim, Args... args){
	assert(dim < order());
	assert(sizeof...(args) <= extent(dim));
	assert(sizeof...(args) < 3); //FOR NOW WORKS ONLY WITH 2 SUBTENSORS

	TensorSlice d;

	d.extents = this->desc_.extents;
	d.extents[dim] = sizeof...(args);
	d.strides = this->desc_.strides;
	d.compute_size();

	std::array<std::size_t, sizeof...(args)> indexes = {
		static_cast<std::size_t>(args)...};

	d.start = this->desc_.start;
	std::sort(indexes.begin(), indexes.end());
	d.start += indexes.front() * this->desc_.strides[dim];

	d.strides[dim] *= (indexes[1] - indexes[0]);

	Tensor<T> res(d, this->elems_);
	if(this->req_grad_){
		res.enable_grad();
		func_variant<T> fn = FunctionId<T>{};

		auto n = std::make_shared<Node<T>>(res);
		n->grad_fn = fn;
		n->set_inputs(*this);

		res.set_node(n);
	}
	return res;
}

template<typename T>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<const T>>
Tensor<T>::dimslices(std::size_t dim, Args... args) const{
	assert(dim < order());
	assert(sizeof...(args) <= extent(dim));
	assert(sizeof...(args) < 3); //FOR NOW WORKS ONLY WITH 2 SUBTENSORS

	TensorSlice d;

	d.extents = this->desc_.extents;
	d.extents[dim] = sizeof...(args);
	d.strides = this->desc_.strides;
	d.compute_size();

	std::array<std::size_t, sizeof...(args)> indexes = {
		static_cast<std::size_t>(args)...};

	d.start = this->desc_.start;
	std::sort(indexes.begin(), indexes.end());
	d.start += indexes.front() * this->desc_.strides[dim];
	
	d.strides[dim] *= (indexes[1] - indexes[0]);

	Tensor<T> res(d, this->elems_);
	if(this->req_grad_){
		res.enable_grad();
		func_variant<T> fn = FunctionId<T>{};

		auto n = std::make_shared<Node<T>>(res);
		n->grad_fn = fn;
		n->set_inputs(*this);

		res.set_node(n);
	}
	return res;
}

template<typename T>
Tensor<T> Tensor<T>::dimslices_arange(std::size_t dim,
		std::size_t from, std::size_t to){
	assert(dim < order());
	assert(from <= to);
	assert(to <= this->extent(dim));

	TensorSlice d;

	d.extents = this->desc_.extents;
	d.extents[dim] = to - from + 1;
	d.strides = this->desc_.strides;
	d.compute_size();

	d.start = this->desc_.start + from * this->desc_.strides[dim];

	Tensor<T> res(d, this->elems_);
	if(this->req_grad_){
		res.enable_grad();
		func_variant<T> fn = FunctionId<T>{};

		auto n = std::make_shared<Node<T>>(res);
		n->grad_fn = fn;
		n->set_inputs(*this);

		res.set_node(n);
	}
	return res;
}

template<typename T>
Tensor<const T> Tensor<T>::dimslices_arange(std::size_t dim,
		std::size_t from, std::size_t to) const{
	assert(dim < order());
	assert(from <= to);
	assert(to <= this->extent(dim));

	TensorSlice d;

	d.extents = this->desc_.extents;
	d.extents[dim] = to - from + 1;
	d.strides = this->desc_.strides;
	d.compute_size();

	d.start = this->desc_.start + from * this->desc_.strides[dim];

	Tensor<T> res(d, this->elems_);
	if(this->req_grad_){
		res.enable_grad();
		func_variant<T> fn = FunctionId<T>{};

		auto n = std::make_shared<Node<T>>(res);
		n->grad_fn = fn;
		n->set_inputs(*this);

		res.set_node(n);
	}
	return res;
}

template<typename T>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<T>>
Tensor<T>::view(Args... args){
	std::size_t args_product = (... * args);
	std::size_t exts_product = std::accumulate(this->desc_.extents.begin(),
			this->desc_.extents.end(), 1, [](std::size_t a,
				std::size_t b) {return a * b;});

	assert(args_product == exts_product);

	std::vector<std::size_t> exts{static_cast<std::size_t>(args)...};
	TensorSlice d{exts};

	Tensor<T> res(d, this->elems_);
	if(this->req_grad_){
		res.enable_grad();
		func_variant<T> fn = FunctionId<T>{};

		auto n = std::make_shared<Node<T>>(res);
		n->grad_fn = fn;
		n->set_inputs(*this);

		res.set_node(n);
	}
	return res;
}

template<typename T>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<const T>>
Tensor<T>::view(Args... args) const{
	std::size_t args_product = (... * args);
	std::size_t exts_product = std::accumulate(this->desc_.extents.begin(),
			this->desc_.extents.end(), 1, [](std::size_t a,
				std::size_t b) {return a * b;});

	assert(args_product == exts_product);

	std::vector<std::size_t> exts{static_cast<std::size_t>(args)...};
	TensorSlice d{exts};

	Tensor<T> res(d, this->elems_);
	if(this->req_grad_){
		res.enable_grad();
		func_variant<T> fn = FunctionId<T>{};

		auto n = std::make_shared<Node<T>>(res);
		n->grad_fn = fn;
		n->set_inputs(*this);

		res.set_node(n);
	}
	return res;
}

//tensor scalar ops
template<typename T>
template<typename F>
Tensor<T>& Tensor<T>::apply(F f){
	for(auto&x : this->elems_) f(x);
	return*this;
}

template<typename T>
Tensor<T>& Tensor<T>::operator+=(const T& val){
	return apply([&](T& a) {a += val;});
}

template<typename T>
Tensor<T>& Tensor<T>::operator-=(const T& val){
	return apply([&](T& a) {a -= val;});
}

template<typename T>
Tensor<T>& Tensor<T>::operator*=(const T& val){
	return apply([&](T& a) {a *= val;});
}

template<typename T>
Tensor<T>& Tensor<T>::operator/=(const T& val){
	return apply([&](T& a) {a /= val;});
}

template<typename T>
Tensor<T>& Tensor<T>::operator%=(const T& val){
	return apply([&](T& a) {a %= val;});
}


//tensor tensor ops
template<typename T>
template<typename M, typename F>
Enable_if<Tensor_type<M>(), Tensor<T>&> Tensor<T>::apply(const M&m, F f){
	assert(same_extents(this->desc_, m.descriptor()));
	for(auto i = begin(), j = m.begin(); i != end(); ++i, ++j)
		f(*i, *j);
	return*this;
}

template<typename T>
template<typename M>
Enable_if<Tensor_type<M>(), Tensor<T>&> Tensor<T>::operator+=(const M&m){
	assert(m.order() == this->order());
	assert(same_extents(desc_, m.descriptor()));
	return apply(m, [&](T& a, const typename M::Value_type&b) {a += b;});
}

template<typename T>
template<typename M>
Enable_if<Tensor_type<M>(), Tensor<T>&> Tensor<T>::operator-=(const M&m){
	assert(m.order() == this->order());
	assert(same_extents(desc_, m.descriptor()));
	return apply(m, [&](T& a, const typename M::Value_type&b) {a -= b;});
}

template<typename T>
template<typename M>
Enable_if<Tensor_type<M>(), Tensor<T>&> Tensor<T>::operator*=(const M&m){
	assert(m.order() == this->order());
	assert(same_extents(desc_, m.descriptor()));
	return apply(m, [&](T& a, const typename M::Value_type&b) {a *= b;});
}

template<typename T>
template<typename M>
Enable_if<Tensor_type<M>(), Tensor<T>&> Tensor<T>::operator/=(const M&m){
	assert(m.order() == this->order());
	assert(same_extents(desc_, m.descriptor()));
	return apply(m, [&](T& a, const typename M::Value_type&b) {a /= b;});
}

template<typename T>
template<typename M>
Enable_if<Tensor_type<M>(), Tensor<T>&> Tensor<T>::operator%=(const M&m){
	assert(m.order() == this->order());
	assert(same_extents(desc_, m.descriptor()));
	return apply(m, [&](T& a, const typename M::Value_type&b) {a %= b;});
}

template<typename T>
template<typename U>
Tensor<U> Tensor<T>::operator-(){
	Tensor<U> res(*this);
	res.apply([&](T& a){a = -a;});

	if(this->req_grad_){
		res.enable_grad();
		func_variant<T> fn = FunctionNeg<T>{};

		auto n = std::make_shared<Node<T>>(res);
		n->grad_fn = fn;
		n->set_inputs(*this);

		res.set_node(n);
	}
	return res;
}

#endif //TENSOR_HPP_
