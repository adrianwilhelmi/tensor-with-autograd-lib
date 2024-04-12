#ifndef TRAITS_HPP_
#define TRAITS_HPP_

#include<type_traits>

#include"declarations.hpp"
#include"tensor.hpp"

template<bool B, typename T = void>
using Enable_if = typename std::enable_if<B,T>::type;

template<typename X, typename Y>
constexpr bool Same(){
	return std::is_same<X,Y>::value;
}

template<typename X, typename Y>
constexpr bool Convertible(){
	return std::is_convertible<X,Y>::value;
}

constexpr bool All() {return true;}

template<typename... Args>
constexpr bool All(bool b, Args... args){
	return b && All(args...);
}

constexpr bool Some() {return false;}

template<typename... Args>
constexpr bool Some(bool b, Args... args){
	return b || Some(args...);
}

struct substitution_failure {};

template<typename T>
struct substitution_succeeded : std::true_type {};

template<>
struct substitution_succeeded<substitution_failure> : std::false_type {};

template<typename M>
struct get_tensor_type_result {
	template<typename T, std::size_t N, typename = Enable_if<(N >= 1)>>
	static bool check(const Tensor<T,N> &m);

	template <typename T, std::size_t N, typename = Enable_if<(N >= 1)>>
	static bool check(const TensorRef<T,N> &m);

	static substitution_failure check(...);

	using type = decltype(check(std::declval<M>()));
};

template<typename T>
struct has_tensor_type : substitution_succeeded<typename get_tensor_type_result<T>::type> {};

template<typename M>
constexpr bool Has_tensor_type(){
	return has_tensor_type<M>::value;
}

template<typename M>
using Tensor_type_result = typename get_tensor_type_result<M>::type;

template<typename M>
constexpr bool Tensor_type(){
	return Has_tensor_type<M>();
}

#endif //TRAITS_HPP_
