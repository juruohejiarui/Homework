all: build/GUI build/CLI

build/CLI:
	cd ./CLI/ && make build && cp ./Game2048-cli ../

build/GUI:
	cd ./GUI/ && qmake ./Game2048.pro && make && cp -r ./Game2048 ../Game2048-gui

clean:
	cd ./CLI/ && make clean
	cd ./GUI/ && make clean
	rm -R ./GUI/Game2048 ./Game2048-gui ./Game2048-cli
