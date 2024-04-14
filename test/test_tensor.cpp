#ifndef TEST_TENSOR_CPP_
#define TEST_TENSOR_CPP_

#include"../include/tensor/storage.hpp"
#include"tensor_lib.hpp"

#include<gtest/gtest.h>

TEST(TensorTest, Constructor){
	Tensor<int,2> t = {
		{1, 2},
		{3, 3},
		{44, 2}
	};
	ASSERT_EQ(t.size(), 6);
	ASSERT_EQ(t.extent(0), 3);
	ASSERT_EQ(t.extent(1), 2);
	ASSERT_EQ(t.order(), 2);
}

TEST(TensorTest, Sharing){
	Tensor<int,2> t1 = {
		{1, 2},
		{3, 3},
		{44, 2}
	};
	Tensor<int,2> t2;
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
	Tensor<int,2> t1 = {
		{1, 2},
		{3, 3},
		{44, 2}
	};
	Tensor<int,1> t2; //= t1.dimslice;
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





#endif
