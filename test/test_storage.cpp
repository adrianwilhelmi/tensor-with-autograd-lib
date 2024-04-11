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
	ASSERT_EQ(storage::same_storage(og, sharee), true);
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

TEST(StorageTest, Iterator){
	Storage<int> og = {3, 4, 12, 1, 0, 44};
	std::sort(og.begin(), og.end());

	EXPECT_EQ(og[0], 0);
	EXPECT_EQ(og[1], 1);
	EXPECT_EQ(og[2], 3);
}

TEST(StorageTest, FromArray){
	std::array<int,4> arr = {7, 2, 5, 4};
	Storage<int> s = storage::from_array(arr);

	ASSERT_EQ(s.size(), arr.size());
	auto sit = s.begin();
	for(auto ait = arr.begin(); ait != arr.end(); ++ait){
		EXPECT_EQ(*sit, *ait);
		++sit;
	}
}

TEST(StorageTest, FromVector){
	std::vector<int> vec = {7, 2, 5, 4};
	Storage<int> s = storage::from_vector(vec);

	ASSERT_EQ(s.size(), vec.size());
	auto sit = s.begin();
	for(auto ait = vec.begin(); ait != vec.end(); ++ait){
		EXPECT_EQ(*sit, *ait);
		++sit;
	}
}

TEST(StorageTest, ToArray){
	Storage<int> s = {1, 4, 4, 7};
	std::array<int, 4> arr = storage::to_array<int,4>(s);

	ASSERT_EQ(s.size(), arr.size());
	auto sit = s.begin();
	for(auto ait = arr.begin(); ait != arr.end(); ++ait){
		EXPECT_EQ(*sit, *ait);
		++sit;
	}
}

TEST(StorageTest, ToVector){
	Storage<int> s = {1, 4, 4, 7};
	std::vector<int> vec = storage::to_vector<int>(s);

	ASSERT_EQ(s.size(), vec.size());
	auto sit = s.begin();
	for(auto ait = vec.begin(); ait != vec.end(); ++ait){
		EXPECT_EQ(*sit, *ait);
		++sit;
	}
}


int main(int argc, char**argv){
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

