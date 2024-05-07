#ifndef FUNCTION_HPP_
#define FUNCTION_HPP_

#include"declarations.hpp"
#include"tensor.hpp"
#include"tensor_ops.hpp"
#include"node.hpp"

#include<variant>
#include<vector>
#include<functional>
#include<iostream>

template<typename Derived, typename T>
class Function{
public:
	void backward(Tensor<T>& grad, node_vector<T>& inputs){
		//CRTP
		static_cast<Derived*>(this)->backward_impl(grad, inputs);
	}
	
private:
	friend Derived;
};

template<typename T>
class FunctionEmpty{
public:
	void backward(Tensor<T>& /*grad*/, node_vector<T>& /*inputs*/) const{}
};

template<typename T>
class FunctionId : public Function<FunctionId<T>, T>{
public:
	using Function<FunctionId<T>,T>::Function;
 
	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		for(auto&node_ptr : inputs){
			if(node_ptr->data.requires_grad()){

				/*
				std::cout << "grad" << std::endl;
				std::cout << grad << std::endl;

				std::cout << "node_ptr->grads() " << std::endl;
				std::cout << node_ptr->grads() << std::endl;

				std::cout << "node_ptr->grads(grad.descriptor()) " << std::endl;
				std::cout << node_ptr->grads(grad.descriptor()) << std::endl;
				*/

				
				node_ptr->grads(grad.descriptor()) += grad;

				/*
				std::cout << "node_ptr->grads(grad.descriptor()) " << std::endl;
				std::cout << node_ptr->grads(grad.descriptor()) << std::endl;

				std::cout << "node_ptr->grads() " << std::endl;
				std::cout << node_ptr->grads() << std::endl;
				*/

				node_ptr->backward();

			}
		}
	}
};

template<typename T>
class FunctionConcat : public Function<FunctionConcat<T>, T>{
public:
	using Function<FunctionConcat<T>,T>::Function;
 
	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		for(auto&node_ptr : inputs){
			if(node_ptr->data.requires_grad()){

				node_ptr->grads += grad(node_ptr->grads.descriptor());
				node_ptr->backward();
			}
		}
	}
};


template<typename T>
class FunctionMul : public Function<FunctionMul<T>, T>{
public:
	using Function<FunctionMul<T>,T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			inputs[0]->grads += inputs[1]->data * grad;
			inputs[0]->backward();
		}
		if(inputs[1]->data.requires_grad()){
			inputs[1]->grads += inputs[0]->data * grad;
			inputs[1]->backward();
		}
	}
};

template<typename T>
class FunctionAdd : public Function<FunctionAdd<T>, T>{
public:
	using Function<FunctionAdd<T>,T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			inputs[0]->grads += grad;
			inputs[0]->backward();
		}
		if(inputs[1]->data.requires_grad()){
			inputs[1]->grads += grad;
			inputs[1]->backward();
		}
	}
};

template<typename T>
class FunctionNeg : public Function<FunctionNeg<T>, T>{
public:
	using Function<FunctionNeg<T>,T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			inputs[0]->grads -= grad;
			inputs[0]->backward();
		}
	}
};

template<typename T>
class FunctionSub : public Function<FunctionSub<T>, T>{
public:
	using Function<FunctionSub<T>,T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			inputs[0]->grads += grad;
			inputs[0]->backward();
		}

		if(inputs[1]->data.requires_grad()){
			inputs[1]->grads -= grad;
			inputs[1]->backward();
		}
	}
};

template<typename T>
class FunctionDiv : public Function<FunctionDiv<T>, T>{
public:
	using Function<FunctionDiv<T>,T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			inputs[0]->grads += inputs[1]->data.pow(T(-1)) * grad;
			inputs[0]->backward();
		}

		if(inputs[1]->data.requires_grad()){
			auto temp = inputs[1]->data.pow(T(-2));
			temp *= inputs[0]->data;
			inputs[1]->grads -= temp * grad;

			inputs[1]->backward();
		}
	}
};

template<typename T>
class FunctionPow : public Function<FunctionPow<T>, T>{
public:
	using Function<FunctionPow<T>,T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			auto t1 = inputs[1]->data - (T)(1);
			auto t2 = inputs[0]->data.pow_(t1);
			inputs[0]->grads += inputs[1]->data * t2 * grad;
			inputs[0]->backward();
		}
		if(inputs[1]->data.requires_grad()){
			auto temp = inputs[0]->data.pow(inputs[1]->data);
			temp *= inputs[0]->data.log();
			inputs[1]->grads += temp * grad;
			inputs[1]->backward();
		}
	}
};

template<typename T>
class FunctionLog : public Function<FunctionLog<T>, T>{
public:
	using Function<FunctionLog<T>,T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			inputs[0]->grads += inputs[1]->data.pow(T(-1)) * grad;
			inputs[0]->backward();
		}
	}
};

template<typename T>
class FunctionExp : public Function<FunctionExp<T>, T>{
public:
	using Function<FunctionExp<T>,T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			inputs[0]->grads += inputs[1]->data * grad;
			inputs[0]->backward();
		}
	}
};

template<typename T>
class FunctionRelu : public Function<FunctionRelu<T>, T>{
public:
	using Function<FunctionRelu<T>,T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			//NOT FINISHED
			//auto temp = inputs[0]->data > T(0);
			inputs[0]->grads += inputs[1]->data * grad;
			inputs[0]->backward();
		}
	}
};

template<typename T>
class FunctionTanh : public Function<FunctionTanh<T>, T>{
public:
	using Function<FunctionTanh<T>,T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){

			Tensor<T> t1 = -inputs[0]->data.tanh().pow(2);
			t1 += 1;
			inputs[0]->grads += t1 * grad;
			inputs[0]->backward();
		}
	}
};

template<typename T>
class FunctionSigmoid : public Function<FunctionSigmoid<T>, T>{
public:
	using Function<FunctionSigmoid<T>,T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			auto temp = inputs[0]->data.sigmoid();

			inputs[0]->grads += temp * (T(1) - temp) * grad;
			inputs[0]->backward();
		}
	}
};

template<typename T>
class FunctionMatmul : public Function<FunctionMatmul<T>, T>{
public:
	using Function<FunctionMatmul<T>, T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			auto temp = inputs[1]->data;

			temp.transpose_();

			/*
			inputs[0]->grads += tensor::matmul(grad, 
					inputs[1]->data.transpose());
			*/

			inputs[0]->grads += tensor::matmul(grad, temp);
			inputs[0]->backward();
		}
		if(inputs[1]->data.requires_grad()){
			auto temp = inputs[0]->data;
			temp.transpose_();

			/*
			inputs[1]->grads += tensor::matmul(
					inputs[0]->data.transpose(),
					grad);
			*/

			inputs[1]->grads += tensor::matmul(temp, grad);
			inputs[1]->backward();
		}
	}
};

namespace function{
	template<typename T>
	bool same_type(const func_variant<T>& f1, const func_variant<T>& f2){
		return std::visit([](auto&&arg1, auto&&arg2){
			return typeid(arg1) == typeid(arg2);
		}, f1, f2);
	}

	template<typename T>
	bool is_type(const func_variant<T>& fv, const std::type_info& type_info){
		return std::visit([&type_info](const auto&arg){
			return typeid(arg) == type_info;
		}, fv);
	}
};

#endif
