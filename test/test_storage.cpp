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

int main(int argc, char**argv){
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

