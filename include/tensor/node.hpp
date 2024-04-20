#ifndef NODE_HPP_
#define NODE_HPP_

#include<variant>

#include"declarations.hpp"
#include"function.hpp"
#include"tensor.hpp"

template<typename T>
struct Node{
	Tensor<T> data;
	func_variant<T> grad_fn;
	std::vector<std::shared_ptr<Node<T>>> inputs;

	Node() : grad_fn(FunctionEmpty<T>{}) {}

	Node(Tensor<T>& res) : grad_fn(FunctionEmpty<T>{}) {
		res.share(this->data);
	}

	template<typename... Args>
	Enable_if<All(Tensor_type<Args>()...), void>
	set_inputs(Args&... args){
		([&]{
			if(args.requires_grad()){
				args.init_grad();
			}
			inputs.push_back(args.get_node());
		}(), ...);
	}

	void backward(){
		std::visit([&](auto& fn){
			fn.backward(this->data.grad(), this->inputs);
		}, this->grad_fn);
	}
};

#endif

