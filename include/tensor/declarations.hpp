#ifndef DECLARATIONS_HPP_
#define DECLARATIONS_HPP_

#include<cstddef>

template<typename T>
class StorageIterator;

template<typename T>
class Storage;

template<std::size_t N>
struct TensorSlice;

template<typename T, std::size_t N>
class TensorRef;

template<typename T, std::size_t N>
class Tensor;

#endif //DECLARATIONS_HPP
