#CXX=g++
CXX=clang++
CXXFLAGS=-Wall -Wextra -Werror -Wno-c11-extensions -pedantic -ggdb -ggdb3 -O3 -g -std=c++17 -Iinclude -I/usr/include/opencv4 -I/usr/include/ -mfma -mavx -mavx2 -pthread
LDLIBS=-lm -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
LDFLAGS= -L/usr/lib/x86_64-linux-gnu -L/usr/lib -pthread
TEST_LDLIBS=$(LDLIBS) -lgtest -lgtest_main -pthread
SFLAGS=-fsanitize=address,undefined

SRC=$(wildcard src/*.cpp)
OBJ=$(SRC:%.cpp=%.o)
EXEC=tsr

TEST_SRC=$(wildcard test/*.cpp)
TEST_OBJ=$(TEST_SRC:%.cpp=%.o)
TEST_EXEC=run_test

EX_SRC=$(wildcard examples/*.cpp)
EX_OBJ=$(EX_SRC:%.cpp=%.o)

all: $(EXEC)

ganalysis: $(OBJ)
	@$(CXX) $(CXXFLAGS) -pg $(OBJ) -o $(EXEC) $(LDFLAGS) $(LDLIBS)
	@./$(EXEC)
	@gprof $(EXEC) gmon.out > analysis.txt
	@rm -f $(OBJ) $(EXEC) gmon.out


analysis: $(OBJ)
	@$(CXX) $(CXXFLAGS) $(OBJ) -o $(EXEC) $(LDFLAGS) $(LDLIBS)
	@valgrind --leak-check=full ./$(EXEC)
	@rm -f $(OBJ) $(EXEC)

example: $(EX_OBJ)
	@$(CXX) $(CXXFLAGS) $(EX_OBJ) -o example $(LDFLAGS) $(LDLIBS)
	@./example
	@rm -f $(EX_OBJ) example

aexample: $(EX_OBJ)
	@$(CXX) $(CXXFLAGS) $(EX_OBJ) -o example $(LDFLAGS) $(LDLIBS)
	@valgrind --leak-check=full ./example
	@rm -f $(EX_OBJ) example

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

install:
	@chmod +x scripts/install_lib.sh
	@scripts/install_lib.sh

%.o: %.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@rm -f $(EXEC) $(TEST_EXEC) $(OBJ) $(TEST_OBJ) example $(EX_OBJ) a.out

.PHONY: all clean test
