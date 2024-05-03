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
		tensor_impl::check_consistency(init);
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

	template<typename T>
	Tensor<T> concat(Tensor<T>& t1, Tensor<T>& t2, std::size_t dim){
		assert(t1.order() == t2.order());
		assert(dim < t1.order());

		for(std::size_t i = 0; i < t1.order(); ++i){
			if(i != dim){
				assert(t1.extent(i) == t2.extent(i) 
						&& "other dims must match");
			}
		}

		std::vector<std::size_t> new_exts(t1.descriptor().extents);
		new_exts[dim] += t2.descriptor().extents[dim];
		TensorSlice desc(new_exts);

		Tensor<T> res(desc);

		for(std::size_t i = 0; i < t1.extent(dim); ++i){
			res.dimslice(dim, i) += t1.dimslice(dim, i);
		}
		for(std::size_t i = 0; i < t2.extent(dim); ++i){
			res.dimslice(dim, i + t1.extent(dim)) += t2.dimslice(dim, i);
		}

		if(t1.requires_grad() || t2.requires_grad()){
			res.enable_grad();

			func_variant<T> fn = FunctionConcat<T>{};
			auto n = std::make_shared<Node<T>>(res);
			n->grad_fn = fn;
			n->set_inputs(t1, t2);

			res.set_node(n);
		}

		return res;
	}

	template<typename T>
	Tensor<T> concat(const Tensor<T>& t1, const Tensor<T>& t2, std::size_t dim){
		assert(t1.order() == t2.order());
		assert(dim < t1.order());

		for(std::size_t i = 0; i < t1.order(); ++i){
			if(i != dim){
				assert(t1.extent(i) == t2.extent(i) 
						&& "other dims must match");
			}
		}

		std::vector<std::size_t> new_exts(t1.descriptor().extents);
		new_exts[dim] += t2.descriptor().extents[dim];
		TensorSlice desc(new_exts);

		Tensor<T> res(desc);

		for(std::size_t i = 0; i < t1.extent(dim); ++i){
			res.dimslice(dim, i) += t1.dimslice(dim, i);
		}
		for(std::size_t i = 0; i < t2.extent(dim); ++i){
			res.dimslice(dim, i + t1.extent(dim)) += t2.dimslice(dim, i);
		}

		return res;
	}

	template<typename T>
	T dot(const Tensor<T>& t1, const Tensor<T>& t2){
		assert(t1.size() == t2.size());
		return std::inner_product(t1.begin(), t1.end(), t2.begin(), T(0));
	}

	template<typename T>
	Tensor<T> matmul(Tensor<T>& t1, Tensor<T>& t2){
		assert(t1.order() == 2);
		assert(t2.order() == 2);
		assert(t1.extent(1) == t2.extent(0));

		Tensor<T> res(t1.extent(0), t2.extent(1));

		for(std::size_t i = 0; i < t1.extent(0); ++i){
			for(std::size_t j = 0; j < t2.extent(1); ++j){
				/*
				res(i,j) = dot(t1.dimslice(0,i),
						t2.dimslice(1,j));
				*/
				T sum = 0;
				for(std::size_t k = 0; k < t1.extent(1); ++k){
					sum += t1(i, k) * t2(k, j);
				}
				res(i, j) = sum;
			}
		}

		if(t1.requires_grad() || t2.requires_grad()){
			res.enable_grad();

			auto n = std::make_shared<Node<T>>(res);
			func_variant<T> fn = FunctionMatmul<T>{};
			n->grad_fn = fn;
			n->set_inputs(t1, t2);

			res.set_node(n);
		}

		return res;
	}

	template<typename T>
	Tensor<T> matmul(const Tensor<T>& t1, const Tensor<T>& t2){
		assert(t1.order() == 2);
		assert(t2.order() == 2);
		assert(t1.extent(1) == t2.extent(0));

		Tensor<T> res(t1.extent(0), t2.extent(1));

		for(std::size_t i = 0; i < t1.extent(0); ++i){
			for(std::size_t j = 0; j < t2.extent(1); ++j){
				T sum = 0;
				for(std::size_t k = 0; k < t1.extent(1); ++k){
					sum += t1(i, k) * t2(k, j);
				}
				res(i, j) = sum;
			}
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
	Tensor<T> res = m * T(-1);
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

	if(a.requires_grad() || b.requires_grad()){
		res.enable_grad();

		func_variant<T> fn = FunctionAdd<T>{};
		auto n = std::make_shared<Node<T>>(res);
		n->grad_fn = fn;
		n->set_inputs(a, b);

		res.set_node(n);
	}

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

	if(a.requires_grad() || b.requires_grad()){
		res.enable_grad();

		func_variant<T> fn = FunctionSub<T>{};
		auto n = std::make_shared<Node<T>>(res);
		n->grad_fn = fn;
		n->set_inputs(a, b);

		res.set_node(n);
	}

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

	if(a.requires_grad() || b.requires_grad()){
		res.enable_grad();
		func_variant<T> fn = FunctionMul<T>{};
		auto n = std::make_shared<Node<T>>(res);
		n->grad_fn = fn;
		n->set_inputs(a, b);

		res.set_node(n);
	}

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

	if(a.requires_grad() || b.requires_grad()){
		res.enable_grad();
		func_variant<T> fn = FunctionDiv<T>{};
		auto n = std::make_shared<Node<T>>(res);
		n->grad_fn = fn;
		n->set_inputs(a, b);

		res.set_node(n);
	}

	return res;
}

template<typename T>
inline Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b){
	Tensor<T> res = a;
	res /= b;
	return res;
}

template<typename T>
inline Tensor<T> operator%(const Tensor<T>& a, const Tensor<T>& b){
	Tensor<T> res = a;
	res %= b;
	return res;
}

#endif //TENSOR_OPS_OPP_


