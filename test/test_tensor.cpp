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






#endif
