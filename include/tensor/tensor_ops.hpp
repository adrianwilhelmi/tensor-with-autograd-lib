#ifndef TENSOR_OPS_HPP_
#define TENSOR_OPS_HPP_

#include"storage.hpp"
#include"tensor.hpp"
#include"node.hpp"
#include"utils/tensor_utils.hpp"

namespace tensor{
	template<typename T>
	bool same_storage(const Tensor<T>&t1, const Tensor<T>&t2){
		return t1.data() == t2.data();
	}

	template<typename T, std::size_t N>
	Tensor<T> from_list(const TensorInitializer<T,N>& init, bool req_grad = false){
		TensorSlice d;
		tensor_impl::derive_extents<T,N>(init, d);
		d.compute_strides();
		Storage<T> elems(d.size);
		tensor_impl::fill_data<T,N>(init, elems, d, 0, 0);

		Tensor<T> res(d, elems);
		if(req_grad){
			res.enable_grad();
		}
		return res;
	}

}; //namespace tensor

template<typename M1, typename M2>
inline Enable_if<Tensor_type<M1>() && Tensor_type<M2>(), bool> operator==(
		const M1&a, const M2&b){
	if(same_extents(a.descriptor(), b.descriptor()))
		return std::equal(a.begin(), a.end(), b.begin());
	return false;
}

template<typename M1, typename M2>
inline Enable_if<Tensor_type<M1>() && Tensor_type<M2>(), bool> operator!=(
		const M1&a, const M2&b){
	return !(a==b);
}


//tensor scalar ops
template<typename T>
Tensor<T> operator+(const Tensor<T>&m, const T&val){
	Tensor<T> res = m;
	res += val;
	return res;
}

template<typename T>
Tensor<T> operator+(const T&val, const Tensor<T>&m){
	Tensor<T> res = m;
	res += val;
	return res;
}

template<typename T>
Tensor<T> operator-(const Tensor<T>&m, const T&val){
	Tensor<T> res = m;
	res -= val;
	return res;
}

template<typename T>
Tensor<T> operator-(const T&val, const Tensor<T>&m){
	Tensor<T> res = -m;
	res += val;
	return res;
}

template<typename T>
Tensor<T> operator*(const Tensor<T>&m, const T&val){
	Tensor<T> res = m;
	res *= val;
	return res;
}

template<typename T>
Tensor<T> operator*(const T&val, const Tensor<T>&m){
	Tensor<T> res = m;
	res *= val;
	return res;
}

template<typename T>
Tensor<T> operator/(const Tensor<T>&m, const T&val){
	Tensor<T> res = m;
	res /= val;
	return res;
}

template<typename T>
Tensor<T> operator/(const T&val, const Tensor<T>&m){
	Tensor<T> res = m;
	res.pow_((T)(-1));
	res *= val;
	return res;
}

template<typename T>
Tensor<T> operator%(const Tensor<T>&m, const T&val){
	Tensor<T> res = m;
	res %= val;
	return res;
}

template<typename T>
Tensor<T> operator%(const T&val, const Tensor<T>&m){
	Tensor<T> res = m;
	res %= val;
	return res;
}

template<typename T>
inline Tensor<T> operator+(Tensor<T>& a, Tensor<T>& b){
	Tensor<T> res = a;
	res += b;

	func_variant<T> fn = FunctionAdd<T>{};
	auto n = std::make_shared<Node<T>>(res);
	n->grad_fn = fn;
	n->set_inputs(a, b);

	res.set_node(n);

	return res;
}

template<typename T>
inline Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b){
	Tensor<T> res = a;
	res += b;
	return res;
}

template<typename T>
inline Tensor<T> operator-(Tensor<T>& a, Tensor<T>& b){
	Tensor<T> res = a;
	res -= b;

	func_variant<T> fn = FunctionSub<T>{};
	auto n = std::make_shared<Node<T>>(res);
	n->grad_fn = fn;
	n->set_inputs(a, b);

	res.set_node(n);

	return res;
}

template<typename T>
inline Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b){
	Tensor<T> res = a;
	res -= b;
	return res;
}

template<typename T>
inline Tensor<T> operator*(Tensor<T>& a, Tensor<T>& b){
	Tensor<T> res = a;
	res *= b;

	func_variant<T> fn = FunctionMul<T>{};
	auto n = std::make_shared<Node<T>>(res);
	n->grad_fn = fn;
	n->set_inputs(a, b);

	res.set_node(n);

	return res;
}

template<typename T>
inline Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b){
	Tensor<T> res = a;
	res *= b;
	return res;
}

template<typename T>
inline Tensor<T> operator/(Tensor<T>& a, Tensor<T>& b){
	Tensor<T> res = a;
	res /= b;

	func_variant<T> fn = FunctionDiv<T>{};
	auto n = std::make_shared<Node<T>>(res);
	n->grad_fn = fn;
	n->set_inputs(a, b);

	res.set_node(n);

	return res;
}

template<typename T>
inline Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b){
	Tensor<T> res = a;
	res /= b;
	return res;
}


#endif //TENSOR_OPS_OPP_


