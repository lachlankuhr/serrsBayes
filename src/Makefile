# Makefile for C++ implementation of fitVoightPeaksSMC
all:
	mkdir -p build
	g++ -I /usr/local/include/eigen3/ fitVoightPeaksSMC.cpp -fopenmp -o build/fitVoightPeaksSMC -g

run: all
	./build/fitVoightPeaksSMC

clean: 
	rm -rf build/*
