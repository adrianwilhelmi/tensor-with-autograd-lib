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
#include<thread>
#include<immintrin.h>

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
	Tensor(const Tensor& x)
		: desc_(x.descriptor().extents), elems_(x.size()){
		this->req_grad_ = false;

		auto xit = x.begin();
		for(auto it = this->begin(); it != this->end(); ++it){
			*it = *xit;
			++xit;
		}

		this->node = nullptr;
	}
	Tensor(const std::size_t size, const T& fill_val = T(0)) 
		: desc_({size}), elems_(size), node(nullptr), req_grad_(false){
		this->desc_.size = size;
		for(auto it = this->begin(); it != this->end; ++it){
			*it = fill_val;
		}
	}
	Tensor(const T val) 
		: desc_({1}), elems_({val}), node(nullptr), req_grad_(false) {}

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

	Tensor<T>& operator=(const T& val){
		std::fill(this->begin(), this->end(), val);
		return*this;
	}

	Tensor(const TensorSlice& d, Storage<T>&e) : desc_(d), req_grad_(false){
		e.share(this->elems_);
		this->node = nullptr;
	}

	Tensor(const TensorSlice& d)
		: desc_(d.extents), elems_(d.size), 
		req_grad_(false) {
		this->node = nullptr;
	}

	Tensor<T> copy_dims() const{
		Storage<T> elems(this->elems_.size());
		//TensorSlice d(this->desc_.extents);
		TensorSlice d(this->desc_);
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
	}

	//indexing
	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(), T&>
	operator()(Args... args);
	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(), const T&>
	operator()(Args... args) const;

	Tensor<T> operator()(const TensorSlice& d) {return {d, this->elems_};}
	Tensor<const T> operator()(const TensorSlice& d) const {return {d, this->elems_};}

	T& item() {return elems_[desc_.start];}
	const T& item() const {return elems_[desc_.start];}

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

	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(),
		Tensor<T>>
	reshape(Args... args);
	template<typename... Args>
	Enable_if<tensor_impl::Requesting_element<Args...>(),
		Tensor<const T>>
	reshape(Args... args) const;

	template<typename U = std::size_t>
	Tensor<U> argmax(){
		TensorSlice d({this->order()});
		Tensor<U> res(d);

		auto mit = this->begin();
		std::vector<std::size_t> max_index(res.size());
		for(auto it = mit; it != this->end(); ++it){
			if(*it > *mit){
				mit = it;
				auto temp = mit.get_index();
				temp[0] -= this->extent(0); // dont ask
				std::copy(temp.begin(), temp.end(),
						res.begin());
			}
		}
		
		return res;
	}

	template<typename U = std::size_t>
	Tensor<const U> argmax() const{
		TensorSlice d({this->order()});
		Tensor<U> res(d);

		auto mit = this->begin();
		std::vector<std::size_t> max_index(res.size());
		for(auto it = mit; it != this->end(); ++it){
			if(*it > *mit){
				mit = it;
				auto temp = mit.get_index();
				temp[0] -= this->extent(0); // dont ask
				std::copy(temp.begin(), temp.end(),
						res.begin());
			}
		}
		
		return res;
	}

	Tensor<T> operator[](std::size_t i) {return dimslice(0, i);}
	Tensor<const T> operator[](std::size_t i) const {return dimslice(0, i);}

	Tensor<T> row(const std::size_t i) {return dimslice(0, i);}
	Tensor<const T> row(const std::size_t i) const {return dimslice(0, i);}

	Tensor<T> col(const std::size_t i) {return dimslice(1, i);}
	Tensor<const T> col(const std::size_t i) const {return dimslice(1, i);}

	Tensor<T> rot180();
	Tensor<const T> rot180() const;

	Tensor<T> rot90();
	Tensor<const T> rot90() const;

	//tensor ops
	//tensor scalar ops
	
	template<typename F>
	Tensor& apply(F f);

	template<typename M, typename F>
	Enable_if<Tensor_type<M>(), Tensor&> apply(const M&m, F f);

	//Tensor& operator=(const T& value);

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
		
		TensorSlice diag_slice(1);
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

	Tensor<const T> diag() const{
		assert(this->order() == 2 && this->extent(0) == this->extent(1));
		
		TensorSlice diag_slice(1);
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




	//MATMUL

	static void mm_block_simd_transposed_b(const Tensor<T>& a, 
			const Tensor<T>& bt, Tensor<T>& res, 
			std::size_t start_row, std::size_t end_row, 
			std::size_t start_col, std::size_t end_col,
			long shared_dim){
		
		for(std::size_t i = start_row; i < end_row; ++i){
			for(std::size_t j = start_col; j < end_col; ++j){
				__m256 sum = _mm256_setzero_ps();
				T partial_sum = 0;

				long k = 0;
				for(k = 0; k < shared_dim - 8; k += 8){
					__m256 vec_a = _mm256_loadu_ps(&a(i,k));
					__m256 vec_b = _mm256_loadu_ps(&bt(j,k));
					sum = _mm256_add_ps(sum, _mm256_mul_ps(
								vec_a, vec_b));
				}

				for(; k < shared_dim; ++k){
					partial_sum += a(i,k) * bt(j,k);
				}

				T buffer[8];
				_mm256_storeu_ps(buffer, sum);

				for(std::size_t l = 0; l < 8; ++l){
					res(i,j) += buffer[l];
				}
				res(i,j) += partial_sum;
			}
		}
	}

	Tensor<T> matmul_optimized(Tensor<T>& other) const{
		if(this->order() != 2 || other.order() != 2)
			throw std::runtime_error("must be 2d matrices");

		if(this->extent(1) != other.extent(0))
			throw std::runtime_error(
					"this->row.size != other.col.size");


		//initializing bt has to be done that way due to memory layout
		Tensor<T> bt(other.extent(1), other.extent(0));
		other.transpose_();

		auto it = other.begin();
		for(auto bit = bt.begin(); bit != bt.end(); ++bit){
			*bit = *it;
			++it;
		}

		other.transpose_();


		std::size_t n = this->extent(0);
		std::size_t p = other.extent(1);
		std::size_t m = this->extent(1);

		Tensor<T> res(n,p);

		const std::size_t num_threads = std::thread::hardware_concurrency();

		std::size_t rows_per_thread = n / num_threads;
		std::size_t extra_rows = n % num_threads;

		if(rows_per_thread <= 1){
			mm_block_simd_transposed_b(std::cref(*this),
					std::cref(bt), 
					std::ref(res), 
					0, n, 0, p, m);
			return res;
		}

		std::vector<std::thread> threads;

		std::size_t start_row = 0;
		for(std::size_t i = 0; i < num_threads; ++i){
			std::size_t end_row = start_row + rows_per_thread + 
				(i < extra_rows ? 1 : 0);

			threads.emplace_back(mm_block_simd_transposed_b, 
					std::cref(*this),
					std::cref(bt),
					std::ref(res),
					start_row, end_row, 0, p, m);
			start_row = end_row;
		}

		for(auto& t : threads){
			t.join();
		}

		return res;
	}




	Tensor<T>& transpose_(const std::size_t d1, const std::size_t d2){
		assert(d1 < this->order() && d2 < this->order());
		std::swap(this->desc_.extents[d1], this->desc_.extents[d2]);
		std::swap(this->desc_.strides[d1], this->desc_.strides[d2]);
		return*this;
	}

	Tensor<T>& transpose_(){
		return transpose_(0,1);
	}

	Tensor<T> transpose(const std::size_t d1, const std::size_t d2){
		assert(d1 < this->order());
		assert(d2 < this->order());

		TensorSlice d = this->desc_;

		std::swap(d.strides[d1], d.strides[d2]);
		std::swap(d.extents[d1], d.extents[d2]);

		Tensor<T> res{d, this->elems_};


		/*
		Tensor<T> res = this->copy_dims();

		auto rit = res.begin();
		auto it = this->begin();
		for(std::size_t i = 0; i < this->size(); ++i){
			*rit = *it;
			++it;
			++rit;
		}

		res.transpose_();
		*/


		if(this->req_grad_){
			res.enable_grad();
			func_variant<T> fn = FunctionConcat<T>{};

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

		return Tensor<T>(d, this->elems_);
	}

	Tensor<const T> transpose() const{
		return transpose(0,1);
	}

	void rot90_(){
		if(order() != 2) throw std::invalid_argument("rot90: must be 2d tensor");

		this->transpose_();

		for(std::size_t i = 0; i < this->extent(1); ++i){
			std::reverse(this->dimslice(1,i).begin(),
					this->dimslice(1,i).end());
		}
	}


	void rot180_(){
		if(order() != 2) throw std::invalid_argument("rot1o0: must be 2d tensor");

		for(std::size_t i = 0; i < this->extent(1); ++i){
			std::reverse(this->dimslice(1,i).begin(),
					this->dimslice(1,i).end());
		}
	}

	//misc
	T sum() const {return std::accumulate(this->begin(), this->end(), T{0});}
	Tensor<T> sum(const long dim){
		if(dim == -1) return sum();

		Tensor<T> res(this->extent(dim));
		for(std::size_t i = 0; i < res.size(); ++i){
			res[i] += this->dimslice(dim, i).sum();
		}

		if(this->requires_grad()){
			res.enable_grad();

			auto n = std::make_shared<Node<T>>(res);
			func_variant<T> fn = FunctionSum<T>{};
			n->grad_fn = fn;
			n->set_inputs(*this);

			res.set_node(n);
		}


		return res;
	}

	Tensor<T> sum(const long dim) const{
		if(dim == -1) return sum();

		Tensor<T> res(this->extent(dim));
		for(std::size_t i = 0; i < (std::size_t)dim; ++i){
			res[i] = this->dimslice(dim, i).sum();
		}

		return res;
	}

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

	T var() const {
		T mean = this->mean();
		T var_sum = 0;
		std::size_t n = 0;
		for(auto it = this->begin(); it != this->end(); ++it, ++n){
			T diff = *it - mean;
			var_sum += diff * diff;
		}
		return n > 0 ? var_sum / n : 0;
	}

	T std() const{
		return std::sqrt(this->var());
	}

	T lp_norm(const float p) const{
		T sum = 0;
		for(auto it = this->begin(); it != this->end(); ++it){
			sum += std::pow(std::abs(*it), p);
		}
		return std::pow(sum, 1 / p);
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

	template<typename U = bool>
	Tensor<U> one_hot(std::size_t num_classes = 0) const{
		if(num_classes == 0){
			num_classes = this->max() + 1;
		}

		Tensor<T> res(this->size(), num_classes);

		auto it = this->begin();
		for(std::size_t i = 0; i < this->size(); ++i){
			res(i, static_cast<std::size_t>(*it) % (num_classes)) = 1;
			++it;
		}
		
		return res;
	}

	Tensor<T> cumsum(){
		Tensor<T> res(*this);
		
		auto rit = res.begin();
		T prev = *rit;
		++rit;
		for(; rit != res.end(); ++rit){
			*rit = *rit + prev;
			prev = *rit;
		}

		return res;
	}

	Tensor<T>& clip(const T low, const T high){
		if(low > high)
			throw std::runtime_error("low > high");

		for(auto it = this->begin(); it != this->end(); ++it){
			if(*it < low){
				*it = low;
			}
			else if(*it > high){
				*it = high;
			}
		}
		return*this;
	}

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

	Tensor<T> log() const{
		Tensor<T> res(*this);
		res.apply([&](T& a) {a = std::log(a);});
		
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

	Tensor<T> relu() {
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

	Tensor<T> softmax() {
		Tensor<T> res(*this);
		T max = res.max();
		res.apply([&](T& a) {a = std::exp(a - max);});
		T sum = res.sum();
		res.apply([&](T&a) {a /= sum;});

		/*
		if(this->req_grad_){
			res.enable_grad();
			func_variant<T> fn = FunctionSoftmax<T>{};
			auto n = std::make_shared<Node<T>>(res);
			n->grad_fn = fn;
			n->set_inputs(*this);

			res.set_node(n);
		}
		*/

		return res;
	}

	Tensor<const T> softmax() const {
		Tensor<const T> res(*this);
		T max = res.max();
		res.apply([&](T& a) {a = std::exp(a - max);});
		T sum = res.sum();
		res.apply([&](T&a) {a /= sum;});

		return res;
	}


	//in-place
	Tensor<T>& pow_(Tensor<T>& exps){
		assert(this->order() == exps.order());
		assert(this->size() == exps.size());
		for(auto i = begin(), j = exps.begin(); i != end(); ++i, ++j)
			*i = std::pow(*i, *j);
		return*this;
	}
	Tensor<T>& pow_(T exp){
		return apply([&](T&a) {a = std::pow(a, exp);});
	}
	Tensor<T>& log_(){
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
			std::cout << "req_grad_ = " << req_grad_ << std::endl;
			std::cout << "(this->node == nullptr) = " << !this->node << std::endl;
			throw std::runtime_error("grad is off");
		}
		return this->node->grads;
	}

	Tensor<T> grad(const TensorSlice& d){
		if(!req_grad_ || !this->node){
			throw std::runtime_error("grad(TensorSlice): grad is off");
		}
		return this->node->grads(d);
	}

	const Tensor<T>& grad(const TensorSlice& d) const{
		if(!req_grad_ || !this->node){
			throw std::runtime_error("grad(): grad is off");
		}
		return this->node->grads(d);
	}

	void zero_grad(){
		if(!req_grad_ || !this->node)
			throw std::runtime_error("zero_grad: grad is off");

		this->node->grads.fill((T)(0));
	}

	void backward_(){
		this->node->backward();
	}

	void backward(){
		if(!node){
			throw std::runtime_error("backward(): grad is off");
		}
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
	:  desc_(x.descriptor().extents), elems_(x.size()), req_grad_(false) {
	static_assert(Convertible<U,T>(), "inconsistent types");
	//std::copy(x.begin(), x.end(), this->begin());

	auto xit = x.begin();
	for(auto it = this->begin(); it != this->end(); ++it){
		*it = *xit;
		++xit;
	}

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
	: desc_(exts...),
	elems_(desc_.size),
	req_grad_(false){

		/*
	if(req_grad){
		this->node = std::make_shared<Node<T>>(*this);
	}
	else{
		this->node = nullptr;
	}
	*/
	this->node = nullptr;
}

//printing
template<typename T>
std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
	if(tensor.order() == 0){
		os << "{" << tensor.item() << "}" << std::endl;
		return os;
	}

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

	//Tensor<const T> res(ts, this->elems_);

	return Tensor<const T>(ts, this->elems_);
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

template<typename T>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<T>>
Tensor<T>::reshape(Args... args) {
	std::size_t args_product = (... * args);
	std::size_t exts_product = std::accumulate(this->desc_.extents.begin(),
			this->desc_.extents.end(), 1, [](std::size_t a,
				std::size_t b) {return a * b;});

	assert(args_product == exts_product);

	std::vector<std::size_t> exts{static_cast<std::size_t>(args)...};
	TensorSlice d{exts};

	//if(!this->desc_.is_contiguous()){
		Tensor<T> res(d);
		auto rit = res.begin();
		auto it = this->begin();
		for(std::size_t i = 0; i < this->size(); ++i){
			*rit = *it;
			++it;
			++rit;
		}

		if(this->req_grad_){
			res.enable_grad();
			func_variant<T> fn = FunctionId<T>{};

			auto n = std::make_shared<Node<T>>(res);
			n->grad_fn = fn;
			n->set_inputs(*this);

			res.set_node(n);
		}

		return res;
	//}
	/*
	else{
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
	*/
}

template<typename T>
template<typename... Args>
Enable_if<tensor_impl::Requesting_element<Args...>(), Tensor<const T>>
Tensor<T>::reshape(Args... args) const {
	std::size_t args_product = (... * args);
	std::size_t exts_product = std::accumulate(this->desc_.extents.begin(),
			this->desc_.extents.end(), 1, [](std::size_t a,
				std::size_t b) {return a * b;});

	assert(args_product == exts_product);

	std::vector<std::size_t> exts{static_cast<std::size_t>(args)...};
	TensorSlice d{exts};

	bool cont = true;
	for(std::size_t i = 1; i < this->desc_.strides.size(); ++i){
		if(this->desc_.strides[i] > this->desc_.strides[i - 1])
			cont = false;
	}

	if(cont){
		Tensor<T> res(d);
		auto it = res.begin();
		for(const auto&elem : *this){
			*it = elem;
			++it;
		}
		return res;
	}
	else{
		Tensor<T> res(d, this->elems_);
		return res;
	}
}

template<typename T>
Tensor<T> Tensor<T>::rot180(){
	if(order() != 2) throw std::invalid_argument("rot180: must be 2d tensor");

	Tensor<T> res = *this;

	for(std::size_t i = 0; i < this->extent(1); ++i){
		std::reverse(res.dimslice(1,i).begin(),
				res.dimslice(1,i).end());
	}

	return res;
}

template<typename T>
Tensor<const T> Tensor<T>::rot180() const{
	if(order() != 2) throw std::invalid_argument("rot180: must be 2d tensor");

	Tensor<T> res = *this;

	for(std::size_t i = 0; i < this->extent(1); ++i){
		std::reverse(res.dimslice(1,i).begin(),
				res.dimslice(1,i).end());
	}

	return res;
}


template<typename T>
Tensor<T> Tensor<T>::rot90(){
	if(order() != 2) throw std::invalid_argument("rot90: must be 2d tensor");

	Tensor<T> res = *this;

	res.transpose_();

	for(std::size_t i = 0; i < res.extent(0); ++i){
		std::reverse(res.dimslice(0,i).begin(),
				res.dimslice(0,i).end());
	}

	return res;
}

template<typename T>
Tensor<const T> Tensor<T>::rot90() const{
	if(order() != 2) throw std::invalid_argument("rot90: must be 2d tensor");

	Tensor<T> res = *this;

	res.transpose_();

	for(std::size_t i = 0; i < this->extent(0); ++i){
		std::reverse(res.dimslice(0,i).begin(),
				res.dimslice(0,i).end());
	}

	return res;
}


//tensor scalar ops
template<typename T>
template<typename F>
Tensor<T>& Tensor<T>::apply(F f){
	for(auto&x : *this) f(x);
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
Enable_if<Tensor_type<M>(), Tensor<T>&> Tensor<T>::apply(const M&other, F f){
	assert(same_extents(this->desc_, other.descriptor()));
	auto j = other.begin();
	for(auto i = begin(); i != end(); ++i){
		f(*i, *j);
		++j;
	}
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
