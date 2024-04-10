#include"../include/tensor/storage.hpp"

#include<gtest/gtest.h>

#include<vector>
#include<string>

TEST(StorageTest, Constructor){
	Storage<int> storage(5);
	ASSERT_EQ(storage.size(), 5);
}

TEST(StorageTest, InitializerList){
	Storage<int> storage = {1, 2, 3, 4, 5};
	std::vector<int> expected = {1, 2, 3, 4, 5};

	ASSERT_EQ(storage.size(), expected.size());
	for(std::size_t i = 0; i < storage.size(); ++i){
		EXPECT_EQ(storage[i], expected[i]);
	}
}

TEST(StorageTest, CopyConstructor){
	Storage<std::string> og = {"c", "a", "b"};
	Storage<std::string> copy(og);
	Storage<std::string> copy2 = og;

	ASSERT_EQ(copy.size(), og.size());
	ASSERT_EQ(copy2.size(), og.size());
	for(std::size_t i = 0; i < og.size(); ++i){
		EXPECT_EQ(copy[i], og[i]);
		EXPECT_EQ(copy2[i], og[i]);
	}
}

/*
TEST(StorageTest, ShareConstructor2){
	Storage<int> og = {4,2,1};
	//Storage<int> sharee(og, true);

	if(true){
		Storage<int> sharee;
		og.share(sharee);

		ASSERT_EQ(og.size(), sharee.size());

		sharee[2] = 2;

		ASSERT_EQ(og.observers(), 2);
	}

	ASSERT_EQ(og[2], 2);
	ASSERT_EQ(og.observers(), 1);
}

TEST(StorageTest, ShareConstructor){
	Storage<int> og = {4,2,1};
	//Storage<int> sharee(og, true);

	Storage<int> sharee;
	og.share(sharee);

	ASSERT_EQ(og.size(), sharee.size());
	for(std::size_t i = 0; i < og.size(); ++i){
		EXPECT_EQ(sharee[i], og[i]);
	}

	ASSERT_EQ(og.observers(), 2);

	og[2] = 2;

	ASSERT_EQ(og == sharee, true);
	ASSERT_EQ(same_storage(og, sharee), true);
}
*/

TEST(StorageTest, MoveConstructor){
	Storage<int> og = {4, 2, 1};
	Storage<int> moved = std::move(og);

	ASSERT_EQ(moved.size(), 3);
	EXPECT_EQ(moved[0], 4);
	EXPECT_EQ(moved[1], 2);
	EXPECT_EQ(moved[2], 1);

	EXPECT_EQ(og.size(), 0);
}

int main(int argc, char**argv){
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

