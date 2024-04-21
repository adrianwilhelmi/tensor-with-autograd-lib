#ifndef DECLARATIONS_HPP_
#define DECLARATIONS_HPP_

#include<cstddef>
#include<variant>
#include<vector>
#include<memory>

template<typename T>
class StorageIterator;

template<typename T>
class Storage;

struct TensorSlice;

template<typename Derived, typename T>
class Function;

template<typename T>
class FunctionEmpty;
template<typename T>
class FunctionId;
template<typename T>
class FunctionConcat;
template<typename T>
class FunctionAdd;
template<typename T>
class FunctionMul;
template<typename T>
class FunctionNeg;
template<typename T>
class FunctionSub;
template<typename T>
class FunctionDiv;
template<typename T>
class FunctionPow;
template<typename T>
class FunctionLog;
template<typename T>
class FunctionExp;
template<typename T>
class FunctionRelu;
template<typename T>
class FunctionTanh;
template<typename T>
class FunctionSigmoid;

template<typename T>
using func_variant = std::variant<
	FunctionEmpty<T>,
	FunctionId<T>,
	FunctionConcat<T>,
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
class Node;

template<typename T>
using node_vector = std::vector<std::shared_ptr<Node<T>>>;

template<typename T>
class Tensor;

#endif //DECLARATIONS_HPP
