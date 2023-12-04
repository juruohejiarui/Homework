CXX=g++ -O2 --std=c++17

BUILD_PATH=./build-cli

.PHONY: build

io-cli.o: io-cli.cpp io-cli.h
	$(CXX) io-cli.cpp -c -o io-cli.o
configuration.o: ../Core/configuration.h ../Core/configuration.cpp ../Core/gamestatepackage.h ../Core/gamestate.h
	$(CXX) ../Core/configuration.cpp -c -o configuration.o
gamestatepackage.o: ../Core/gamestatepackage.cpp ../Core/gamestatepackage.h ../Core/gamestate.h
	$(CXX) ../Core/gamestatepackage.cpp -c -o gamestatepackage.o
gamestate.o: ../Core/gamestate.cpp ../Core/gamestate.h
	$(CXX) ../Core/gamestate.cpp -c -o gamestate.o
gameboard-cli.o: io-cli.h gameboard-cli.h gameboard-cli.cpp ../Core/configuration.h ../Core/gamestatepackage.h ../Core/gamestate.h io-cli.h
	$(CXX) gameboard-cli.cpp -c -o gameboard-cli.o
main-cli.o: gameboard-cli.h ../Core/gamestate.h ../Core/gamestatepackage.h ../Core/configuration.h
	$(CXX) main-cli.cpp -c -o main-cli.o

build: configuration.o gamestatepackage.o gamestate.o gameboard-cli.o main-cli.o io-cli.o
	$(CXX) configuration.o gamestatepackage.o gamestate.o gameboard-cli.o io-cli.o main-cli.o -o ./Game2048-cli

clean:
	rm ./*.o ./Game2048-cli
