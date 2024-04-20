#ifndef NODE_HPP_
#define NODE_HPP_

#include<variant>
#include<iostream>

#include"declarations.hpp"
#include"function.hpp"
#include"tensor.hpp"

template<typename T>
struct Node{
	Tensor<T> data;
	Tensor<T> grad;
	func_variant<T> grad_fn;
	std::vector<std::shared_ptr<Node<T>>> inputs;

	Node() : grad_fn(FunctionEmpty<T>{}) {}

	void init_grad(){
		Tensor<T> g(data.descriptor());
		g.fill(T(0));
		std::cout << "g" << std::endl;
		std::cout << g << std::endl;
		this->grad = g;
	}

	Node(Tensor<T>& res) : grad_fn(FunctionEmpty<T>{}) {
		res.share(this->data);
		if(res.requires_grad()){
			init_grad();
		}
	}

	template<typename... Args>
	Enable_if<All(Tensor_type<Args>()...), void>
	set_inputs(Args&... args){
		([&]{
			if(args.requires_grad()){
				args.init_grad();
				inputs.push_back(args.get_node());
			}
		}(), ...);
	}

	void backward(){
		std::visit([&](auto& fn){
			fn.backward(this->grad, this->inputs);
		}, this->grad_fn);
	}
};

#endif

