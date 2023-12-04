all: build/GUI build/CLI

build/CLI:
	cd ./CLI/ && make build && cp ./Game2048-cli ../

build/GUI:
	cd ./GUI/ && make && cp -r ./Game2048.app ../Game2048-gui.app
