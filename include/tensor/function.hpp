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
#include<stdexcept>

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
				node_ptr->grads(grad.descriptor()) += grad;
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
			auto temp = inputs[0]->data;
			temp.exp_();
			inputs[0]->grads += temp * grad;
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
			auto temp = inputs[0]->data;
			temp.relu_();

			auto tit = temp.begin();
			auto git = grad.begin();
			auto rit = inputs[0]->grads.begin();
			for(auto tit = temp.begin(); 
					tit != temp.end(); ++ tit){
				if(*tit > 0){
					*rit = *git;
				}
				++rit;
				++git;
			}

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
class FunctionSoftmax : public Function<FunctionSoftmax<T>, T>{
public:
	using Function<FunctionSoftmax<T>,T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			inputs[0]->grads += grad;
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
			//auto temp = inputs[1]->data;
			//temp.transpose_();

			inputs[0]->grads += 
				grad.matmul_optimized_transposed_b(inputs[1]->data);

			//inputs[0]->grads += tensor::matmul(grad, temp);
			//inputs[0]->grads += grad.matmul_optimized(temp);
			inputs[0]->backward();
		}
		if(inputs[1]->data.requires_grad()){
			/*
			auto temp = inputs[0]->data;
			temp.transpose_();
			*/

			inputs[0]->data.transpose_();
			inputs[1]->grads += inputs[0]->data.matmul_optimized(grad);
			inputs[0]->data.transpose_();

			//inputs[1]->grads += tensor::matmul(temp, grad);
			//inputs[1]->grads += temp.matmul_optimized(grad);
			inputs[1]->backward();
		}
	}
};

template<typename T>
class FunctionConv2d : public Function<FunctionConv2d<T>, T>{
public:
	using Function<FunctionConv2d<T>, T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			auto temp = inputs[1]->data;
			for(std::size_t i = 0; i < temp.extent(0); ++i){
				temp.dimslice(0,i).rot180_();
			}

			inputs[0]->grads += tensor::conv2d(grad, temp);
			inputs[0]->backward();
		}
		if(inputs[1]->data.requires_grad()){
			inputs[1]->grads += tensor::conv2d(inputs[0]->data, 
						grad);
			inputs[1]->backward();
		}
	}
};

template<typename T>
class FunctionMaxPooling : public Function<FunctionMaxPooling<T>, T>{
public:
	using Function<FunctionMaxPooling<T>, T>::Function;

	//FunctionMaxPooling(std::kernel_size, std::size_t stride)


	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){

			const std::size_t num_channels = inputs[0]->data.extent(0);
			const std::size_t input_height = inputs[0]->data.extent(1);
			const std::size_t input_width = inputs[0]->data.extent(2);

			const std::size_t output_height = grad.extent(1);
			const std::size_t output_width = grad.extent(2);

			const std::size_t kernel_size = (input_width * (output_height - 1) - input_height * (output_width - 1)) / (output_height - output_width);

			const std::size_t stride = (input_height - kernel_size) / (output_height - 1);

			for(std::size_t c = 0; c < num_channels; ++c){
				for(std::size_t i = 0; i < output_height; ++i){
					for(std::size_t j = 0; j < output_width; ++j){
						T max_val = std::numeric_limits<T>::lowest();
						long long max_i = 0, max_j = 0;
						for(std::size_t ki = 0; ki < kernel_size; ++ki){
							for(std::size_t kj = 0; kj < kernel_size; ++kj){
								long long ii = i * stride + ki;
								long long jj = j * stride + kj;

								if(ii < (long long)input_height && jj <(long long)input_width){
									if(inputs[0]->data(c, ii, jj) > max_val){
										max_val = inputs[0]->data(c, ii, jj);
										max_i = ii;
										max_j = jj;
									}
								}
							}
						}
						inputs[0]->grads(c, max_i, max_j) += grad(c, i ,j);
					}
				}
				
			}

			//inputs[0]->grads += 
			inputs[0]->backward();
		}
	}
};

template<typename T>
class FunctionCrossEntropy : public Function<FunctionCrossEntropy<T>, T>{
public:
	using Function<FunctionCrossEntropy<T>, T>::Function;

	void backward_impl(Tensor<T>& /*grad*/, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			inputs[0]->grads += (inputs[0]->data.softmax() - inputs[1]->data);
			inputs[0]->backward();
		}
	}
};

template<typename T>
class FunctionSum : public Function<FunctionSum<T>, T>{
public:
	using Function<FunctionSum<T>, T>::Function;

	void backward_impl(Tensor<T>& grad, node_vector<T>& inputs){
		if(inputs[0]->data.requires_grad()){
			std::size_t shape = inputs[0]->data.order();
			long common = -1;
			for(std::size_t i = 0; i < shape; ++i){
				if(grad.size() == 
						inputs[0]->data.extent(i)){
					common = i;
				}
			}

			if(common == -1){
				throw std::runtime_error("no common extents");
			}

			for(std::size_t i = 0; i < grad.size(); ++i){

				Tensor<T> temp(inputs[0]->grads.dimslice(common,i).descriptor());

				//temp.fill(grad[i]);
				temp = grad(i);

				inputs[0]->grads.dimslice(common, i) += temp;
			}
			
			inputs[0]->backward();
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
