CXX := g++
CXX_ARG := -std=c++14
all: class_inherit
clean:
	rm -rf ./*.o ./class_inherit
class_inherit.o: class_inherit.h class_inherit.cpp
	$(CXX) $(CXX_ARG) class_inherit.cpp -c
main.o: main.cpp class_inherit.h
	$(CXX) $(CXX_ARG) main.cpp -c
class_inherit: main.o class_inherit.o
	$(CXX) main.o class_inherit.o -o class_inherit