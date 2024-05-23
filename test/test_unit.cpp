#ifndef TEST_UNIT_CPP
#define TEST_UNIT_CPP

#include<gtest/gtest.h>

int main(int argc, char**argv){
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

#endif
