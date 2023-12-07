all: build/GUI build/CLI
	rm default.config default.config.state

build/CLI:
ifeq ($(OS),Windows_NT)
	cd ./CLI/ && make build && cp ./Game2048-cli.exe ../
else ifeq ($(shell uname),Darwin)
	cd ./CLI/ && make build && cp ./Game2048-cli ../
else 
	cd ./CLI/ && make build && cp ./Game2048-cli ../
endif

build/GUI:
ifeq ($(OS),Windows_NT)
	cd ./GUI/ && qmake ./Game2048-gui.pro && make && copy -r ./Game2048.exe ../Game2048-gui.exe
else ifeq ($(shell uname),Darwin)
	cd ./GUI/ && qmake ./Game2048-gui.pro && make && cp -r ./Game2048-gui.app ../Game2048-gui.app
else 
	cd ./GUI/ && qmake ./Game2048-gui.pro && make && cp -r ./Game2048-gui ../Game2048-gui
endif

clean:
	cd ./GUI/ && make clean && rm Makefile
	cd ./CLI/ && make clean
ifeq ($(OS),Windows_NT)
	rm -R ./GUI/Game2048.exe ./Game2048-gui.exe ./Game2048-cli.exe
else ifeq ($(shell uname),Darwin)
	rm -R ./Game2048-gui.app ./Game2048-cli ./GUI/Game2048-gui.app
else 
	rm -R ./GUI/Game2048 ./Game2048-gui ./Game2048-cli
endif
	rm default.config default.config.state