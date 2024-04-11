#ifndef TRAITS_HPP_
#define TRAITS_HPP_

#include<type_traits>

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

#endif //TRAITS_HPP_
