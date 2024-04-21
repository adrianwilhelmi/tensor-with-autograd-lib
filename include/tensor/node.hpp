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
	Tensor<T> grads;
	func_variant<T> grad_fn;
	std::vector<std::shared_ptr<Node<T>>> inputs;

	Node() : grad_fn(FunctionEmpty<T>{}) {}

	void init_grad(){
		Tensor<T> g(this->data.descriptor());
		g.fill(T(0));
		this->grads = g;
	}

	Node(Tensor<T>& res) : grads(res.size()), grad_fn(FunctionEmpty<T>{}) {
		res.share(this->data);
		if(res.requires_grad()){
			//this->init_grad();
			this->grads = res.copy_dims();
		}
	}

	template<typename... Args>
	Enable_if<All(Tensor_type<Args>()...), void>
	set_inputs(Args&... args){
		([&]{
			if(args.get_node() == nullptr){
				auto n = std::make_shared<Node<T>>(args);
				args.set_node(n);
			}
		 	inputs.push_back(args.get_node());
		}(), ...);
	}

	void backward(){
		std::visit([&](auto& fn){
			fn.backward(this->grads, this->inputs);
		}, this->grad_fn);
	}
};

#endif

