SHELL=/bin/bash
CWD := $(shell pwd)
CXXFLAGS := -g -pg -std=c++11 -Wall -Wpedantic
INCLUDES := -I $(CWD)/npy_array/include -I $(CWD)/npy_array/src/
LIBS := -L $(CWD)/npy_array/lib -lnpy_array -L /usr/local/lib -lboost_regex

all:
	g++ $(CXXFLAGS) $(INCLUDES) imslic.cpp -o imslic $(LIBS) -Wl,-rpath=$(CWD)/npy_array/lib/
	#time $(CWD)/imslic
	g++ -std=c++11 -g $(INCLUDES) imslic_test.cpp -o imslic_test $(LIBS) -lgtest -Wl,-rpath=$(CWD)/npy_array/lib/
	./imslic_test