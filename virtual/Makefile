CFLAGS 	?= -std=gnu99 -Wall
VIRT_RT ?= runtime.h
OBJ = build/workspace.o build/parspl.o build/smain.o build/kernel.o build/virtual_main.o

.PHONY: clean all

all: run

clean:
	rm -rf build

build:
	mkdir -p $@

virtual:
	echo TODO

build/%.o: %.c $(VIRT_RT) | build
	gcc $(CFLAGS) -g -c $< -o $@

build/virtual_main: $(OBJ)
	#prerequisites: $?
	#output: $@
	gcc $(CFLAGS) -lpthread $^ -o $@

run: build/virtual_main
	./build/virtual_main
