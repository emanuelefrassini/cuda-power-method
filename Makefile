CXX ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
NVCCFLAGS ?= -O3 -std=c++17

.PHONY: all cpu gpu test test-gpu clean
all: cpu
cpu: bin/power_cpu
gpu: bin/power_cuda

bin/power_cpu: src/power_cpu.cpp include/power_common.hpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ src/power_cpu.cpp

bin/power_cuda: src/power_cuda.cu include/power_common.hpp
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -Iinclude -o $@ src/power_cuda.cu

test: cpu
	python3 tests/test_power.py --backend cpu

test-gpu: gpu
	python3 tests/test_power.py --backend cuda

clean:
	rm -rf bin
