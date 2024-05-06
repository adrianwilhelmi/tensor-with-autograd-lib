CXX=g++
#CXX=clang++
CXXFLAGS=-Wall -Wextra -Werror -pedantic -ggdb -ggdb3 -O3 -g -std=c++17 -Iinclude -I/usr/include/opencv4
LDLIBS=-lm -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
LDFLAGS= -L/usr/lib/x86_64-linux-gnu
TEST_LDLIBS=$(LDLIBS) -lgtest -lgtest_main -pthread
SFLAGS=-fsanitize=address,undefined

SRC=$(wildcard src/*.cpp)
OBJ=$(SRC:%.cpp=%.o)
EXEC=tsr

TEST_SRC=$(wildcard test/*.cpp)
TEST_OBJ=$(TEST_SRC:%.cpp=%.o)
TEST_EXEC=run_test

all: $(EXEC)

$(EXEC): $(OBJ)
	@$(CXX) $(CXXFLAGS) $(OBJ) -o $(EXEC) $(LDFLAGS) $(LDLIBS)
	@./$(EXEC)
	@rm -f $(OBJ) $(EXEC)

test: $(TEST_OBJ)
	@$(CXX) $(CXXFLAGS) $(TEST_OBJ) -o $(TEST_EXEC) $(LDFLAGS) $(TEST_LDLIBS)
	@./$(TEST_EXEC)
	@make clean

atest: $(TEST_OBJ)
	@$(CXX) $(CXXFLAGS) $(TEST_OBJ) -o $(TEST_EXEC) $(LDFLAGS) $(TEST_LDLIBS)
	@valgrind --leak-check=full ./$(TEST_EXEC)
	@make clean

analysis: $(OBJ)
	@$(CXX) $(CXXFLAGS) $(OBJ) -o $(EXEC) $(LDFLAGS) $(LDLIBS)
	@valgrind --leak-check=full ./$(EXEC)
	@rm -f $(OBJ) $(EXEC)

%.o: %.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@rm -f $(EXEC) $(TEST_EXEC) $(OBJ) $(TEST_OBJ) a.out

.PHONY: all clean test
