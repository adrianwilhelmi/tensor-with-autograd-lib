#ifndef TENSOR_OPS_HPP_
#define TENSOR_OPS_HPP_

#include<chrono>
#include<random>

#include<opencv2/opencv.hpp>

#include"declarations.hpp"
#include"storage.hpp"
#include"tensor.hpp"
#include"node.hpp"
#include"utils/tensor_utils.hpp"

namespace tensor{
	template<typename T, typename U>
	bool same_storage(const Tensor<T>&t1, const Tensor<U>&t2){
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
	Tensor<T> from_image(const std::string& filepath){
		cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
		if(image.empty()){
			throw std::runtime_error("couldnt open or find the image");
		}

		cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

		if(std::is_same<T, float>::value){
			image.convertTo(image, CV_32FC3, 1/255.0f);
		}
		else if(std::is_same<T, int>::value){
			image.convertTo(image, CV_32SC3);
		}
		else{
			throw std::runtime_error("must be float or int");
		}

		Tensor<T> tensor_img(3, image.rows, image.cols);

		for(auto y = 0; y < image.rows; ++y){
			for(auto x = 0; x < image.cols; ++x){
				//implement
				cv::Vec3f color = image.at<cv::Vec3f>(y,x);

				tensor_img(0, y, x) = static_cast<T>(color[2]);
				tensor_img(1, y, x) = static_cast<T>(color[1]);
				tensor_img(2, y, x) = static_cast<T>(color[0]);
			}
		}

		return tensor_img;
	}

	template<typename T>
	Tensor<T> from_video(const std::string& filepath){
		cv::VideoCapture cap(filepath);
		if(!cap.isOpened()){
			throw std::runtime_error("couldnt open or find the video file");
		}

		auto num_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
		auto frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		auto frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);

		Tensor<T> tensor_vid(3, num_frames, frame_height, frame_width);

		cv::Mat frame;
		int frame_idx = 0;
		while(cap.read(frame)){
			if(std::is_same<T, float>::value){
				frame.convertTo(frame, CV_32FC3, 1/255.0f);
			}
			else if(std::is_same<T, int>::value){
				frame.convertTo(frame, CV_32SC3);
			}

			cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

			for(auto y = 0; y < frame.rows; ++y){
				for(auto x = 0; x < frame.cols; ++x){
					cv::Vec3f color = frame.at<cv::Vec3f>(y,x);
					tensor_vid(0, frame_idx, y, x) = color[0];
					tensor_vid(1, frame_idx, y, x) = color[1];
					tensor_vid(2, frame_idx, y, x) = color[2];
				}
			}
			++frame_idx;
		}

		return tensor_vid;
	}

	template<typename T, typename... Exts>
	Tensor<T> zeros(Exts... exts){
		Tensor<T> res(exts...);
		std::fill(res.begin(), res.end(), T{0});
		return res;
	}

	template<typename T, typename... Exts>
	Tensor<T> ones(const Exts... exts){
		Tensor<T> res(exts...);
		std::fill(res.begin(), res.end(), T{1});
		return res;
	}

	template<typename T, typename... Exts>
	Tensor<T> random_normal(const T mean, const T stddev, const Exts... exts){
		Tensor<T> res(exts...);

		auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::mt19937 gen(seed);
		std::normal_distribution<T> dist(mean, stddev);

		for(auto it = res.begin(); it != res.end(); ++it){
			*it = dist(gen);
		}

		return res;
	}

	template<typename T>
	Tensor<T> random_normal(const T mean, const T stddev, const Tensor<T>&t){
		Tensor<T> res = t.copy_dims();

		auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::mt19937 gen(seed);
		std::normal_distribution<T> dist(mean, stddev);

		for(auto it = res.begin(); it != res.end(); ++it){
			*it = dist(gen);
		}

		return res;
	}

	template<typename T, typename... Exts>
	Tensor<T> random_bernoulli(const double p, Exts... exts){
		Tensor<T> res(exts...);

		auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::mt19937 gen(seed);
		std::bernoulli_distribution dist(p);

		for(auto it = res.begin(); it != res.end(); ++it){
			*it = dist(gen);
		}

		return res;
	}

	template<typename T>
	Tensor<T> random_bernoulli(const double p, const Tensor<T>&t){
		Tensor<T> res = t.copy_dims();

		auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::mt19937 gen(seed);
		std::bernoulli_distribution dist(p);

		for(auto it = res.begin(); it != res.end(); ++it){
			*it = dist(gen);
		}

		return res;
	}

	template<typename T, typename... Exts>
	Tensor<T> random_uniform(const T min, const T max, const Exts... exts){
		Tensor<T> res(exts...);

		auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::mt19937 gen(seed);
		std::uniform_real_distribution<T> dist(min, max);

		for(auto it = res.begin(); it != res.end(); ++it){
			*it = dist(gen);
		}

		return res;
	}

	template<typename T>
	Tensor<T> random_uniform(const T min, const T max, const Tensor<T>& t){
		Tensor<T> res = t.copy_dims();

		auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::mt19937 gen(seed);
		std::uniform_real_distribution<T> dist(min, max);

		for(auto it = res.begin(); it != res.end(); ++it){
			*it = dist(gen);
		}

		return res;
	}

	template<typename T = int, typename... Exts>
	Tensor<T> randint(const int min, const int max, const Exts... exts){
		Tensor<T> res(exts...);

		auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::mt19937 gen(seed);
		std::uniform_int_distribution<T> dist(min, max);

		for(auto it = res.begin(); it != res.end(); ++it){
			*it = dist(gen);
		}

		return res;
	}

	template<typename T = int>
	Tensor<T> randint(const int min, const int max, const Tensor<T>& t){
		Tensor<T> res = t.copy_dims();

		auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::mt19937 gen(seed);
		std::uniform_int_distribution<T> dist(min, max);

		for(auto it = res.begin(); it != res.end(); ++it){
			*it = dist(gen);
		}

		return res;
	}


	template<typename U, typename T = std::size_t>
	Tensor<T> random_multinomial(const Tensor<U>& probs, 
			std::size_t num_samples, bool replacement = true){
		if(probs.sum() < U(0.99)){
			throw std::runtime_error("probabilities doesnt sum to 1");
		}
		auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::mt19937 gen(seed);
		
		std::vector<double> weights(probs.size());

		auto wit = weights.begin();
		for(auto it = probs.begin(); it != probs.end(); ++it){
			*wit = *it;
			++wit;
		}

		Storage<std::size_t> counts(num_samples);

		for(std::size_t i = 0; i < num_samples; ++i){
			std::discrete_distribution<> dist(weights.begin(), 
							weights.end());
			int index = dist(gen);
			counts[i] = index;

			if(!replacement)
				weights[index] = 0;
		}

		std::vector<std::size_t> exts = {num_samples};
		TensorSlice d(exts);

		return Tensor<T>(d, counts);
	}

	template<typename T>
	Tensor<T> eye(const std::size_t n){
		Tensor<T> res(n, n);
		res.diag().fill(T(1));
		return res;
	}

	template<typename T>
	Tensor<T> arange(const T start, 
			const T end,
			const T step = 1){
		if(step == 0){
			throw std::invalid_argument("step = 0");
		}

		std::size_t size = static_cast<std::size_t>(
				std::ceil(static_cast<double>(end - start) / step));
		TensorSlice d({size});

		Storage<T> elems(size);

		T val = (T) start;
		for(auto& elem : elems){
			elem = val;
			val += (T) step;
		}

		return {d, elems};
	}

	template<typename T>
	void to_image(const Tensor<T>& tensor, const std::string& filepath){
		if(tensor.order() != 3 || tensor.extent(0) != 3){
			throw std::runtime_error("this tensor doesnt represent an image");
		}

		std::size_t height = tensor.extent(1);
		std::size_t width = tensor.extent(2);

		cv::Mat image(height, width, CV_32FC3);
		//auto type = std::is_same<T,float>::value ? CV_32FC3 : CV_8UC3;

		//auto mul = std::is_same<T,float>::value ? 255.0f : 1;

		for(std::size_t y = 0; y < height; ++y){
			for(std::size_t x = 0; x < width; ++x){
				cv::Vec3f color = {
					tensor(2,y,x),
					tensor(1,y,x),
					tensor(0,y,x)
				};
				image.at<cv::Vec3f>(y,x) = color;
			}
		}

		if(std::is_same<T, float>::value){
			image.convertTo(image, CV_8UC3, 255.0f);
		}
		else if(std::is_same<T,int>::value){
			image.convertTo(image, CV_8UC3);
		}

		cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
		cv::imwrite(filepath, image);
	}

	template<typename T>
	void to_video(const Tensor<T>& tensor, const std::string& filepath){
		if(tensor.order() != 4 || tensor.extent(0) != 3){
			throw std::runtime_error("this tensor doesnt represent a video");
		}

		auto num_frames = tensor.extent(0);
		auto height = tensor.extent(2);
		auto width = tensor.extent(3);

		cv::VideoWriter writer(filepath, cv::VideoWriter::fourcc('M','J','P','G'), 30, cv::Size(width, height), true);

		for(auto i = 0; i < num_frames; ++i){
			cv::Mat frame(height, width, CV_32FC3);
			for(auto y = 0; y < height; ++y){
				for(auto x = 0; x < width; ++x){
					cv::Vec3f color = {
						tensor(2, i, y, x),
						tensor(1, i, y, x),
						tensor(0, i, y, x)
					};
					frame.at<cv::Vec3f>(y,x) = color;
				}
			}

			if(std::is_same<T, float>::value){
				frame.convertTo(frame, CV_8UC3, 255.0);
			}
			else if(std::is_same<T,int>::value){
				frame.convertTo(frame, CV_8UC3);
			}
			cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
			writer.write(frame);
		}
	}


	template<typename T, typename U = bool>
	Tensor<U> one_hot(const Tensor<T>& t, std::size_t num_classes = 0){
		if(num_classes == 0){
			num_classes = t.max() + 1;
		}

		Tensor<T> res(t.size(), num_classes);

		auto it = t.begin();
		for(std::size_t i = 0; i < t.size(); ++i){
			res(i, static_cast<std::size_t>(*it) % (num_classes)) = 1;
			++it;
		}
		
		return res;
	}


	template<typename T>
	bool nearly_equal(const Tensor<T>& t1, const Tensor<T>& t2){
		double epsilon = 0.02;
		auto it2 = t2.begin();
		if(same_extents(t1.descriptor(), t2.descriptor())){
			for(auto it1 = t1.begin(); it1 != t1.end(); ++it1){
				if(std::abs(*it1 - *it2) > epsilon)
					return false;
				++it2;
			}
			return true;
		}
		return false;
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
	Tensor<T> mean(Tensor<T>& t){
		Tensor<T> res = t.mean();
		
		if(t.requires_grad()){
			res.enable_grad();
			auto n = std::make_shared<Node<T>>(res);
			func_variant<T> fn = FunctionMean<T>{};
			n->grad_fn = fn;
			n->set_inputs(t);

			res.set_node(n);
		}

		return res;
	}

	template<typename T>
	Tensor<T> mean(const Tensor<T>& t){
		return t.mean();
	}

	template<typename T>
	T dot(const Tensor<T>& t1, const Tensor<T>& t2){
		assert(t1.size() == t2.size());
		return std::inner_product(t1.begin(), t1.end(), t2.begin(), T(0));
	}


	template<typename T>
	Tensor<T> matmul(Tensor<T>& t1, Tensor<T>& t2){
		if(t1.order() != 2 || t2.order() != 2)
			throw std::runtime_error("must be a 2d matrix");
		if(t1.extent(1) != t2.extent(0))
			throw std::runtime_error("cant multiply these matrices");

		Tensor<T> bt(t2.extent(1), t2.extent(0));
		bt.transpose_();

		auto bit = bt.begin();
		for(auto it = t2.begin(); it != t2.end(); ++it){
			*bit = *it;
			++bit;
		}

		bt.transpose_();

		Tensor<T> res = t1.matmul_optimized_transposed_b(bt);

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
		if(t1.order() != 2 || t2.order() != 2)
			throw std::runtime_error("must be a 2d matrix");
		if(t1.extent(1) != t2.extent(0))
			throw std::runtime_error("cant multiply these matrices");

		Tensor<T> bt(t2.extent(1), t2.extent(0));
		bt.transpose_();

		auto bit = bt.begin();
		for(auto it = t2.begin(); it != t2.end(); ++it){
			*bit = *it;
			++bit;
		}

		bt.transpose_();

		return t1.matmul_optimized_transposed_b(bt);
	}

	template<typename T>
	Tensor<T> matmul_classic(Tensor<T>& t1, Tensor<T>& t2){
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
	Tensor<T> matmul_classic(const Tensor<T>& t1, const Tensor<T>& t2){
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


	template<typename T>
	Tensor<T> conv2d(Tensor<T>& input, 
			Tensor<T>& kernel,
			const std::size_t stride = 1,
		       	const std::size_t padding = 1){
		const std::size_t out_height = (input.extent(1) - kernel.extent(1) + 2 * padding) / stride + 1;
		const std::size_t out_width = (input.extent(2) - kernel.extent(2) + 2 * padding) / stride + 1;

		Tensor<T> res(input.extent(0), out_height, out_width);

		auto num_channels = input.extent(0);

		for(std::size_t c = 0; c < num_channels; ++c){
			for(std::size_t i = 0; i < out_height; ++i){
				for(std::size_t j = 0; j < out_width; ++j){
					T sum = 0;
					for(std::size_t ki = 0; ki < kernel.extent(1); ++ki){
						for(std::size_t kj = 0; kj < kernel.extent(2); ++kj){
							long long ii = i * stride + ki - padding;
							long long jj = j * stride + kj - padding;
							if(ii >= 0 && ii < (long long)input.extent(1) && jj >= 0 && jj < (long long)input.extent(2)){
								sum += input(c, ii, jj) * kernel(c, ki, kj);
							}
						}
					}
					res(c, i, j) = sum;
				}
			}
		}

		if(input.requires_grad() || kernel.requires_grad()){
			res.enable_grad();

			auto n = std::make_shared<Node<T>>(res);
			func_variant<T> fn = FunctionConv2d<T>{};
			n->grad_fn = fn;
			n->set_inputs(input, kernel);

			res.set_node(n);
		}

		return res;
	}

	template<typename T>
	Tensor<T> conv2d(const Tensor<T>& input, 
			const Tensor<T>& kernel,
			const std::size_t stride = 1,
		       	const std::size_t padding = 1){
		const std::size_t out_height = (input.extent(1) - kernel.extent(1) + 2 * padding) / stride + 1;
		const std::size_t out_width = (input.extent(2) - kernel.extent(2) + 2 * padding) / stride + 1;

		Tensor<T> res(input.extent(0), out_height, out_width);

		auto num_channels = input.extent(0);

		for(std::size_t c = 0; c < num_channels; ++c){
			for(std::size_t i = 0; i < out_height; ++i){
				for(std::size_t j = 0; j < out_width; ++j){
					T sum = 0;
					for(std::size_t ki = 0; ki < kernel.extent(1); ++ki){
						for(std::size_t kj = 0; kj < kernel.extent(2); ++kj){
							long long ii = i * stride + ki - padding;
							long long jj = j * stride + kj - padding;
							if(ii >= 0 && ii < (long long)input.extent(1) && jj >= 0 && jj < (long long)input.extent(2)){
								sum += input(c, ii, jj) * kernel(c, ki, kj);
							}
						}
					}
					res(c, i, j) = sum;
				}
			}
		}

		return res;
	}
	
	template<typename T>
	Tensor<T> max_pooling(Tensor<T>& input, std::size_t kernel_size,
			std::size_t stride){
		if(input.order() != 3) throw std::invalid_argument("max_pooling: must be a 3d tensor");

		const std::size_t num_channels = input.extent(0);
		const std::size_t input_height = input.extent(1);
		const std::size_t input_width = input.extent(2);

		const std::size_t out_height = (input_height - kernel_size) / stride + 1;
		const std::size_t out_width = (input_width - kernel_size) / stride + 1;

		Tensor<T> res(num_channels, out_height, out_width);

		for(std::size_t c = 0; c < num_channels; ++c){
			for(std::size_t i = 0; i < out_height; ++i){
				for(std::size_t j = 0; j < out_width; ++j){
					auto submat = input.dimslice(0, c).
							dimslices_range(0, i * stride, 
								i * stride + kernel_size - 1).
							dimslices_range(1, j * stride,
								j * stride + kernel_size - 1);

					res(c,i,j) = submat.max();
				}
			}
		}

		if(input.requires_grad()){
			res.enable_grad();

			auto n = std::make_shared<Node<T>>(res);
			func_variant<T> fn = FunctionMaxPooling<T>{};
			n->grad_fn = fn;
			n->set_inputs(input);

			res.set_node(n);
		}

		return res;
	}

	template<typename T>
	Tensor<T> max_pooling(const Tensor<T>& input, std::size_t kernel_size,
			std::size_t stride = 1){
		if(input.order() != 3) throw std::invalid_argument("max_pooling: must be a 3d tensor");

		const std::size_t num_channels = input.extent(0);
		const std::size_t input_height = input.extent(1);
		const std::size_t input_width = input.extent(2);

		const std::size_t out_height = (input_height - kernel_size) / stride + 1;
		const std::size_t out_width = (input_width - kernel_size) / stride + 1;

		Tensor<T> res(num_channels, out_height, out_width);

		for(std::size_t c = 0; c < num_channels; ++c){
			for(std::size_t i = 0; i < out_height; ++i){
				for(std::size_t j = 0; j < out_width; ++j){
					auto submat = input.dimslice(0, c).
							dimslices_range(0, i * stride, 
								i * stride + kernel_size - 1).
							dimslices_range(1, j * stride,
								j * stride + kernel_size - 1);

					res(c,i,j) = submat.max();
				}
			}
		}

		return res;
	}


	template<typename T>
	Tensor<T> avg_pooling(Tensor<T>& input, std::size_t kernel_size,
			std::size_t stride){
		if(input.order() != 3) throw std::invalid_argument("max_pooling: must be a 3d tensor");

		const std::size_t num_channels = input.extent(0);
		const std::size_t input_height = input.extent(1);
		const std::size_t input_width = input.extent(2);

		const std::size_t out_height = (input_height - kernel_size) / stride + 1;
		const std::size_t out_width = (input_width - kernel_size) / stride + 1;

		Tensor<T> res(num_channels, out_height, out_width);

		for(std::size_t c = 0; c < num_channels; ++c){
			for(std::size_t i = 0; i < out_height; ++i){
				for(std::size_t j = 0; j < out_width; ++j){
					auto submat = input.dimslice(0, c).
							dimslices_range(0, i * stride, 
								i * stride + kernel_size - 1).
							dimslices_range(1, j * stride,
								j * stride + kernel_size - 1);

					res(c,i,j) = submat.sum();
					res(c,i,j) /= kernel_size * kernel_size;
				}
			}
		}

		if(input.requires_grad()){
			res.enable_grad();

			auto n = std::make_shared<Node<T>>(res);
			func_variant<T> fn = FunctionConv2d<T>{};
			n->grad_fn = fn;
			n->set_inputs(input);

			res.set_node(n);
		}

		return res;
	}

	template<typename T>
	Tensor<T> avg_pooling(const Tensor<T>& input, std::size_t kernel_size,
			std::size_t stride){
		if(input.order() != 3) throw std::invalid_argument("max_pooling: must be a 3d tensor");

		const std::size_t num_channels = input.extent(0);
		const std::size_t input_height = input.extent(1);
		const std::size_t input_width = input.extent(2);

		const std::size_t out_height = (input_height - kernel_size) / stride + 1;
		const std::size_t out_width = (input_width - kernel_size) / stride + 1;

		Tensor<T> res(num_channels, out_height, out_width);

		for(std::size_t c = 0; c < num_channels; ++c){
			for(std::size_t i = 0; i < out_height; ++i){
				for(std::size_t j = 0; j < out_width; ++j){
					auto submat = input.dimslice(0, c).
							dimslices_range(0, i * stride, 
								i * stride + kernel_size - 1).
							dimslices_range(1, j * stride,
								j * stride + kernel_size - 1);

					res(c,i,j) = submat.sum();
					res(c,i,j) /= kernel_size * kernel_size;
				}
			}
		}

		return res;
	}

	template<typename T>
	Tensor<T> cross_entropy(Tensor<T>& logits, Tensor<T>& targets){
		if(logits.order() != targets.order())
			throw std::invalid_argument("cross_entropy: inconsistent orders");
		
		for(std::size_t i = 0; i < logits.order(); ++i){
			if(logits.extent(i) != targets.extent(i))
				throw std::invalid_argument("cross_entropy: inconsistent extents");
		}


		/*
		constexpr T epsilon = 1e-8;
		Tensor<T> t = logits.softmax();
		t.clip_(epsilon, (T)1 - epsilon);
		t.log_();
		t *= targets;

		Tensor<T> res = -(t.sum() / logits.extent(0));
		*/
		

		constexpr T epsilon = 1e-8;

		std::size_t batch_size = logits.extent(0);

		Tensor<T> res(batch_size);

		for(std::size_t i = 0; i < batch_size; ++i){
			Tensor<T> t = logits[i].softmax();
			t.clip_(epsilon, (T)1 - epsilon);
			t.log_();
			t *= targets[i];

			res[i] += -(t.sum() / batch_size);
		}


		if(logits.requires_grad()){
			res.enable_grad();
			auto n = std::make_shared<Node<T>>(res);
			func_variant<T> fn = FunctionCrossEntropy<T>{};
			n->grad_fn = fn;
			n->set_inputs(logits, targets);
			res.set_node(n);
		}

		return res;
	}

	template<typename T>
	Tensor<T> cross_entropy(const Tensor<T>& logits, const Tensor<T>& targets){
		if(logits.order() != 2 || targets.order() != 2)
			throw std::invalid_argument("cross_entropy: must be a 2d tensor");
 		if(logits.extent(0) != targets.extent(0) || logits.extent(1) != targets.extent(1)) 
			throw std::invalid_argument("cross_entropy: inconsistent extents");

		constexpr T epsilon = 1e-8;
		Tensor<T> t = logits.softmax();
		t.clip_(epsilon, (T)1 - epsilon);
		t.log_();
		t *= targets;
		
		Tensor<T> res = -(t.sum() / logits.extent(0));

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


