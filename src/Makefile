# Makefile for C++ implementation of fitVoightPeaksSMC
all:
	mkdir -p build
	g++ -I /usr/local/include/eigen3/ fitVoightPeaksSMC.cpp -o build/fitVoightPeaksSMC

run: all
	./build/fitVoightPeaksSMC

clean: 
	rm -rf build/*
