

struct tensor_slice{
	tensor_slice(); //= default;
	tensor_slice(std::size_t dims, std::size_t s, std::initializer_list<std::size_t> exts);
	tensor_slice(std::size_t dims, std::size_t s, std::initializer_list<std::size_t> exts, std::initializer_list<std::size_t> strs);

	template<typename N>
	tensor_slice(const std::array<std::size_t, N> &exts);

	template<typename... Dims>
	tensor_slice(Dims... dims);

	template<typename... Dims>
	std::size_t operator()(Dims... dims) const;

	template<typename N>
	std::size_t offset(const std::array<std::size_t, N> &pos) const;

	void clear();

	std::size_t dims;
	std::size_t size;
	std::size_t start;
	std::size_t extents[dims];
	std::size_t strides[dims];
};

tensor_slice::tensor_slice() : dims{1}, size{1}, start{0} {
	*extents = 1;
	*strides = 1;
}

tensor_slice::tensor_slice(std::size_t dims, std::size_t s,
		std::initializer_list<std::size_t> exts)
	: start{s}{
		assert(exts.size() == N);
		std::copy(exts.begin(), extents.end(), extents.begin());
