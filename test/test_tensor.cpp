#ifndef TEST_TENSOR_CPP_
#define TEST_TENSOR_CPP_

#include"../include/tensor/storage.hpp"
#include"tensor_lib.hpp"

#include<gtest/gtest.h>

TEST(TensorTest, Constructor){
	Tensor<int> t = tensor::from_list<int,2>({
		{1, 2},
		{3, 3},
		{44, 2}
	});
	ASSERT_EQ(t.size(), 6);
	ASSERT_EQ(t.extent(0), 3);
	ASSERT_EQ(t.extent(1), 2);
	ASSERT_EQ(t.order(), 2);
}

TEST(TensorTest, Sharing){
	Tensor<int> t1 = tensor::from_list<int,2>({
		{1, 2},
		{3, 3},
		{44, 2}
	});
	Tensor<int> t2;
	//t1.share(t2);
	t2 = t1;

	ASSERT_EQ(t1.size(), t2.size());
	ASSERT_EQ(t1.extent(0), t1.extent(0));
	ASSERT_EQ(t1.extent(1), t1.extent(1));
	ASSERT_EQ(t1.order(), t1.order());

	t1(1,1) = 14;

	EXPECT_EQ(t2(1,1), 14);

	//ASSERT_TRUE(storage::same_storage(t1.storage(), t2.storage()));
	ASSERT_TRUE(tensor::same_storage(t1, t2));
}

TEST(TensorTest, Dimslice){
	Tensor<int> t1 = tensor::from_list<int,2>({
		{1, 2},
		{3, 3},
		{44, 2}
	});
	Tensor<int> t2; //= t1.dimslice;
	//t1.share(t2);
	t2 = t1.dimslice(1, 1);

	ASSERT_EQ(t1.dimslice(1,1).size(), t2.size());
	ASSERT_EQ(t1.dimslice(1,1).extent(0), t2.extent(0));
	ASSERT_EQ(t1.dimslice(1,1).order(), t2.order());

	t2(1) = 14;

	EXPECT_EQ(t2(1), t1(1,1));

	//ASSERT_TRUE(storage::same_storage(t1.storage(), t2.storage()));
	ASSERT_TRUE(tensor::same_storage(t1.dimslice(1,1), t2));
}

TEST(TensorTest, AddOperatorAutograd){
	Tensor<int> t1 = tensor::from_list<int,2>({
		{4, 2},
		{2, 3},
	}, true);

	Tensor<int> t2 = tensor::from_list<int,2>({
		{1, 2},
		{3, 3},
	}, true);

	Tensor<int> t3 = t1 + t2;

	EXPECT_EQ(t3(0,0), 5);
	EXPECT_EQ(t3(0,1), 4);
	EXPECT_EQ(t3(1,0), 5);
	EXPECT_EQ(t3(1,1), 6);

	t3.backward();

	Tensor<int> res = tensor::from_list<int,2>({
		{1, 1},
		{1, 1}
	}, false);

	ASSERT_EQ(t1.grad(), res);
	ASSERT_EQ(t2.grad(), res);
}

TEST(TensorTest, MulOperatorAutograd){
	Tensor<int> t1 = tensor::from_list<int,2>({
		{4, 2},
		{2, 3},
	}, true);

	Tensor<int> t2 = tensor::from_list<int,2>({
		{1, 2},
		{3, 3},
	}, true);

	Tensor<int> t3 = t1 * t2;

	EXPECT_EQ(t3(0,0), 4);
	EXPECT_EQ(t3(0,1), 4);
	EXPECT_EQ(t3(1,0), 6);
	EXPECT_EQ(t3(1,1), 9);

	t3.backward();

	ASSERT_EQ(t1.grad(), t2);
	ASSERT_EQ(t2.grad(), t1);
}

TEST(TensorTest, SubOperatorAutograd){
	Tensor<int> t1 = tensor::from_list<int,2>({
		{4, 2},
		{2, 3},
	}, true);

	Tensor<int> t2 = tensor::from_list<int,2>({
		{1, 2},
		{3, 3},
	}, true);

	Tensor<int> t3 = t1 - t2;

	EXPECT_EQ(t3(0,0), 3);
	EXPECT_EQ(t3(0,1), 0);
	EXPECT_EQ(t3(1,0), -1);
	EXPECT_EQ(t3(1,1), 0);

	t3.backward();

	Tensor<int> res = tensor::from_list<int,2>({
		{1, 1},
		{1, 1}
	}, false);

	ASSERT_EQ(t1.grad(), res);
	ASSERT_EQ(t2.grad(), -res);
}

TEST(TensorTest, 0DimTensor){
	Tensor<double> vec = tensor::from_list<double,1>(
		{1, 5, 4, 8, 9, 4}
		, false);

	auto s1 = vec.dimslice(0,1);
	auto s2 = vec.dimslice(0,4);

	s1.enable_grad();
	s2.enable_grad();

	auto s = s1 * s2;

	s.backward();

	EXPECT_EQ(s1.grad(), s2);
	EXPECT_EQ(s2.grad(), s1);
}

TEST(TensorTest, TensorImageConversionFloat){
	const std::string four_path = "./test/photos/four.png";
	Tensor<float> four = tensor::from_image<float>(four_path);

	const std::string troll_path = "./test/photos/trll.jpeg";
	Tensor<float> troll = tensor::from_image<float>(troll_path);

	const std::string four_path2 = "./test/photos/four2.png";
	tensor::to_image<float>(four, four_path2);

	const std::string troll_path2 = "./test/photos/troll2.png";
	tensor::to_image<float>(troll, troll_path2);

	Tensor<float> four2 = tensor::from_image<float>(four_path2);

	Tensor<float> troll2 = tensor::from_image<float>(troll_path2);


	ASSERT_TRUE(tensor::nearly_equal(troll2, troll));
	ASSERT_TRUE(tensor::nearly_equal(four2, four));
}

TEST(TensorTest, RandomConstructorAndSoftmax){
	Tensor<double> rnd = tensor::random_normal(0.0, 1.0, 4, 4);
	Tensor<double> rnds = rnd.softmax();

	ASSERT_TRUE(rnds.sum() >= 0.99);
}

TEST(TensorTest, TransposeTest){
	Tensor<double> t2 = tensor::from_list<double,2>({
		{1, 5, 2},
		{1, 4, 16},
		{1, 10, 32},
		{31, 12, 3},
	}, true);

	Tensor<double> t2tok = tensor::from_list<double,2>({
		{1, 1, 1, 31},
		{5, 4, 10, 12},
		{2, 16, 32, 3}
	}, true);

	auto t2t = t2.transpose();

	ASSERT_TRUE(t2tok == t2t);
}


TEST(TensorTest, TranspositionReshapeGradient){
	Tensor<double> t3 = tensor::from_list<double,2>({
		{1, 2, 3},
		{15, 44, 3},
		{31, 12, 3},
		{31, 12, 3},
	}, false);

	Tensor<double> t2 = tensor::from_list<double,2>({
		{1, 5, 2},
		{1, 4, 16},
		{1, 10, 32},
		{31, 12, 3},
	}, false);

	Tensor<double> t3dt(2,4,3);

	t3dt.dimslice(0,0) += t2;
	t3dt.dimslice(0,1) += t2 * t3;

	Tensor<double> t3d = t3dt.transpose(1, 2);

	t3d.enable_grad();

	Tensor<double> twos = t3d.copy_dims();
	twos.fill(2.0);
	twos.enable_grad();

	Tensor<double> t3d2 = t3d * twos;


	Tensor<double> t2d = t3d2.reshape(
			t3d2.extent(0) * t3d2.extent(1),
			t3d2.extent(2)
			);

	Tensor<double> t2dt = t2d.transpose();

	t2dt.backward();

	ASSERT_EQ(t3d.grad(), twos);
	ASSERT_EQ(t3d, twos.grad());
}


#endif
