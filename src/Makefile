# Makefile for C++ implementation of fitVoightPeaksSMC
all: program
	

program: gpu.o:
	mkdir -p build
	nvcc -o build/program -I /usr/local/include/eigen3/ fitVoightPeaksSMC.cpp moveGPU.cu --expt-relaxed-constexpr

gpu.o:
	nvcc -c moveGPU.cu

run: all
	./build/fitVoightPeaksSMC

clean: 
	rm -rf build/*
