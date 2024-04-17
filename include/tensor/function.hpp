#ifndef FUNCTION_HPP_
#define FUNCTION_HPP_

#include"tensor.hpp"

#include<variant>
#include<vector>
#include<functional>

template<typename Derived, typename T>
class Function{
public:
	Function(std::reference_wrapper<Tensor<T>> a,
			std::reference_wrapper<Tensor<T>> b)
		: a(a), b(b){
		if(a.get().requires_grad()) a.get().init_grad();
		if(b.get().requires_grad()) b.get().init_grad();
	}

	void backward(const Tensor<T>& grad){
		//CRTP
		static_cast<Derived*>(this)->backward_impl(grad);
	}

protected:
	std::reference_wrapper<Tensor<T>> a;
	std::reference_wrapper<Tensor<T>> b;
private:
	friend Derived;
};

class FunctionEmpty{
public:
	template<typename T>
	void backward(const Tensor<T>&/*grad*/){}
};

template<typename T>
class FunctionMul : public Function<FunctionMul<T>, T>{
public:
	using Function<FunctionMul<T>,T>::Function;

	void backward_impl(const Tensor<T>& grad){
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

template<typename T>
class FunctionAdd : public Function<FunctionAdd<T>, T>{
public:
	using Function<FunctionAdd<T>,T>::Function;

	void backward_impl(const Tensor<T>& grad){
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

template<typename T>
class FunctionNeg : public Function<FunctionNeg<T>, T>{
public:
	using Function<FunctionNeg<T>,T>::Function;

	void backward_impl(const Tensor<T>&grad){
		if(this->a.get().requires_grad()) {
			this->a.get().grad() -= grad;
			this->a.get().backward_();
		}
	}
};

template<typename T>
class FunctionSub : public Function<FunctionSub<T>, T>{
public:
	using Function<FunctionSub<T>,T>::Function;

	void backward_impl(const Tensor<T>&grad){
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

template<typename T>
class FunctionDiv : public Function<FunctionDiv<T>, T>{
public:
	using Function<FunctionDiv<T>,T>::Function;

	void backward_impl(const Tensor<T>&grad){
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

template<typename T>
class FunctionPow : public Function<FunctionPow<T>, T>{
public:
	using Function<FunctionPow<T>,T>::Function;

	void backward_impl(const Tensor<T>&grad){
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

template<typename T>
class FunctionLog : public Function<FunctionLog<T>, T>{
public:
	using Function<FunctionLog<T>,T>::Function;

	void backward_impl(const Tensor<T>&grad){
		if(this->a.get().requires_grad()) {
			this->a.get().grad() += this->b.get().pow(T(-1)) * grad;
			this->a.get().backward_();
		}
	}
};

template<typename T>
class FunctionExp : public Function<FunctionExp<T>, T>{
public:
	using Function<FunctionExp<T>,T>::Function;

	void backward_impl(const Tensor<T>&grad){
		if(this->a.get().requires_grad()) {
			this->a.get().grad() += this->b.get() * grad;
			this->a.get().backward_();
		}
	}
};

template<typename T>
class FunctionRelu : public Function<FunctionRelu<T>, T>{
public:
	using Function<FunctionRelu<T>,T>::Function;

	void backward_impl(const Tensor<T>&grad){
		if(this->a.get().requires_grad()) {
			auto temp = this->a.get().pow(0);
			this->a.get().grad() += this->b.get() * grad;
			this->a.get().backward_();
		}
	}
};


template<typename T>
class FunctionTanh : public Function<FunctionTanh<T>, T>{
public:
	using Function<FunctionTanh<T>,T>::Function;

	void backward_impl(const Tensor<T>&grad){
		if(this->a.get().requires_grad()) {
			Tensor<T> t1 = -this->b.get().pow(2);
			t1 += 1;
			this->a.get().grad() += t1 * grad;
			this->a.get().backward_();
		}
	}
};

template<typename T>
class FunctionSigmoid : public Function<FunctionSigmoid<T>, T>{
public:
	using Function<FunctionSigmoid<T>,T>::Function;

	void backward_impl(const Tensor<T>&grad){
		if(this->a.get().requires_grad()) {
			auto temp = this->b.get() - T(1);
			this->a.get().grad() += this->b.get() * temp * grad;
			this->a.get().backward_();
		}
	}
};

template<typename T>
using func_variant = std::variant<
	FunctionEmpty,
	FunctionAdd<T>,
	FunctionMul<T>,
	FunctionNeg<T>,
	FunctionSub<T>,
	FunctionDiv<T>,
	FunctionPow<T>,
	FunctionLog<T>,
	FunctionExp<T>,
	FunctionRelu<T>,
	FunctionTanh<T>,
	FunctionSigmoid<T>
>;

template<typename T>
bool same_func_type(const func_variant<T>& f1, const func_variant<T>& f2){
	return std::visit([](auto&&arg1, auto&&arg2){
		return typeid(arg1) == typeid(arg2);
	}, f1, f2);
}

template<typename T>
bool is_func_type(const func_variant<T>& fv, const std::type_info& type_info){
	return std::visit([&type_info](const auto&arg){
		return typeid(arg) == type_info;
	}, fv);
}

#endif
