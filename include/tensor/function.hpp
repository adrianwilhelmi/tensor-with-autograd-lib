#ifndef FUNCTION_HPP_
#define FUNCTION_HPP_

#include"tensor.hpp"

#include<variant>
#include<vector>
#include<functional>

template<typename Derived, typename T, std::size_t N>
class Function{
public:
	Function(std::reference_wrapper<Tensor<T,N>> a,
			std::reference_wrapper<Tensor<T,N>> b)
		: a(a), b(b){
		if(a.get().requires_grad()) a.get().init_grad();
		if(b.get().requires_grad()) b.get().init_grad();
	}

	void backward(Tensor<T,N>grad){
		//CRTP
		static_cast<Derived*>(this)->backward_impl(grad);
	}

protected:
	std::reference_wrapper<Tensor<T,N>> a;
	std::reference_wrapper<Tensor<T,N>> b;
private:
	friend Derived;
};

class FunctionEmpty{
public:
	template<typename T, std::size_t N>
	void backward(const Tensor<T,N>&/*grad*/){}
};

template<typename T, std::size_t N>
class FunctionMul : public Function<FunctionMul<T,N>, T, N>{
public:
	using Function<FunctionMul<T,N>,T,N>::Function;

	void backward_impl(const Tensor<T,N>& grad){
		if(this->a.get().requires_grad()) {
			this->a.get().grad() += this->b.get() * grad;
			this->a.get().backward_();
		}

		if(this->b.get().requires_grad()) {
			this->b.get().grad() += this->a.get() * grad;
			this->b.get().backward_();
		}
	}
};

template<typename T, std::size_t N>
class FunctionAdd : public Function<FunctionAdd<T,N>, T, N>{
public:
	using Function<FunctionAdd<T,N>,T,N>::Function;

	void backward_impl(const Tensor<T,N>& grad){
		if(this->a.get().requires_grad()) {
			this->a.get().grad() += grad;
			this->a.get().backward_();
		}

		if(this->b.get().requires_grad()) {
			this->b.get().grad() += grad;
			this->b.get().backward_();
		}
	}
};

template<typename T, std::size_t N>
class FunctionNeg : public Function<FunctionNeg<T,N>, T, N>{
public:
	using Function<FunctionNeg<T,N>,T,N>::Function;

	void backward_impl(const Tensor<T,N>&grad){
		if(this->a.get().requires_grad()) {
			this->a.get().grad() -= grad;
			this->a.get().backward_();
		}
	}
};

template<typename T, std::size_t N>
class FunctionSub : public Function<FunctionSub<T,N>, T, N>{
public:
	using Function<FunctionSub<T,N>,T,N>::Function;

	void backward_impl(const Tensor<T,N>&grad){
		if(this->a.get().requires_grad()) {
			this->a.get().grad() += grad;
			this->a.get().backward_();
		}

		if(this->b.get().requires_grad()) {
			this->b.get().grad() -= grad;
			this->b.get().backward_();
		}
	}
};

template<typename T, std::size_t N>
class FunctionDiv : public Function<FunctionDiv<T,N>, T, N>{
public:
	using Function<FunctionDiv<T,N>,T,N>::Function;

	void backward_impl(const Tensor<T,N>&grad){
		if(this->a.get().requires_grad()) {
			this->a.get().grad() += this->b.get().pow(T(-1)) * grad;
			this->a.get().backward_();
		}

		if(this->b.get().requires_grad()) {
			auto temp = this->b.get().pow(T(-2));
			temp *= this->a.get();
			this->b.get().grad() -= temp * grad;
			this->b.get().backward_();
		}
	}
};

template<typename T, std::size_t N>
class FunctionPow : public Function<FunctionPow<T,N>, T, N>{
public:
	using Function<FunctionPow<T,N>,T,N>::Function;

	void backward_impl(const Tensor<T,N>&grad){
		if(this->a.get().requires_grad()) {
			auto t1 = this->b.get() - (T)(1);
			auto t2 = this->a.get().pow_(t1);
			this->a.get().grad() += this->b.get() * t2 * grad;
			this->a.get().backward_();
		}
		if(this->b.get().requires_grad()){
			auto temp = this->a.get().pow(this->b.get());
			temp *= this->a.get().log();
			this->b.get().grad() += temp * grad;
			this->b.get().backward_();
		}
	}
};

template<typename T, std::size_t N>
class FunctionLog : public Function<FunctionLog<T,N>, T, N>{
public:
	using Function<FunctionLog<T,N>,T,N>::Function;

	void backward_impl(const Tensor<T,N>&grad){
		if(this->a.get().requires_grad()) {
			this->a.get().grad() += this->b.get().pow(T(-1)) * grad;
			this->a.get().backward_();
		}
	}
};

template<typename T, std::size_t N>
class FunctionExp : public Function<FunctionExp<T,N>, T, N>{
public:
	using Function<FunctionExp<T,N>,T,N>::Function;

	void backward_impl(const Tensor<T,N>&grad){
		if(this->a.get().requires_grad()) {
			this->a.get().grad() += this->b.get() * grad;
			this->a.get().backward_();
		}
	}
};

template<typename T, std::size_t N>
class FunctionRelu : public Function<FunctionRelu<T,N>, T, N>{
public:
	using Function<FunctionRelu<T,N>,T,N>::Function;

	void backward_impl(const Tensor<T,N>&grad){
		if(this->a.get().requires_grad()) {
			auto temp = this->a.get().pow(0);
			this->a.get().grad() += this->b.get() * grad;
			this->a.get().backward_();
		}
	}
};


template<typename T, std::size_t N>
class FunctionTanh : public Function<FunctionTanh<T,N>, T, N>{
public:
	using Function<FunctionTanh<T,N>,T,N>::Function;

	void backward_impl(const Tensor<T,N>&grad){
		if(this->a.get().requires_grad()) {
			Tensor<T,N> t1 = -this->b.get().pow(2);
			t1 += 1;
			this->a.get().grad() += t1 * grad;
			this->a.get().backward_();
		}
	}
};

template<typename T, std::size_t N>
class FunctionSigmoid : public Function<FunctionSigmoid<T,N>, T, N>{
public:
	using Function<FunctionSigmoid<T,N>,T,N>::Function;

	void backward_impl(const Tensor<T,N>&grad){
		if(this->a.get().requires_grad()) {
			auto temp = this->b.get() - T(1);
			this->a.get().grad() += this->b.get() * temp * grad;
			this->a.get().backward_();
		}
	}
};

template<typename T, std::size_t N>
using func_variant = std::variant<
	FunctionEmpty,
	FunctionAdd<T,N>,
	FunctionMul<T,N>,
	FunctionNeg<T,N>,
	FunctionSub<T,N>,
	FunctionDiv<T,N>,
	FunctionPow<T,N>,
	FunctionLog<T,N>,
	FunctionExp<T,N>,
	FunctionRelu<T,N>,
	FunctionTanh<T,N>,
	FunctionSigmoid<T,N>
>;

#endif
