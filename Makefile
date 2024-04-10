CXX=g++
CXXFLAGS=-Wall -Wextra -Werror -pedantic -ggdb -ggdb3 -O3 -g -std=c++17
LDFLAGS=-Iinclude
LDLIBS=-lm
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
	@$(CXX) $(OBJ) -o $(EXEC) $(LDLIBS)
	@./$(EXEC)
	@rm -f $(OBJ) $(EXEC)

test: $(TEST_OBJ) $(OBJ)
	@$(CXX) $(TEST_OBJ) $(filter-out -src/main.o, $(OBJ)) -o $(TEST_EXEC) $(TEST_LDLIBS)
	@./$(TEST_EXEC)
	@make clean

analysis: $(OBJ)
	@$(CXX) $(OBJ) -o $(EXEC) $(LDLIBS)
	@valgrind --leak-check=full ./$(EXEC)
	@rm -f $(OBJ) $(EXEC)

%.o: %.cpp
	@$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

clean:
	@rm -f $(EXEC) $(TEST_EXEC) $(OBJ) $(TEST_OBJ) a.out

.PHONY: all clean test
