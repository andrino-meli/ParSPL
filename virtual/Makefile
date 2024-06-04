CFLAGS 	?= -std=gnu99 -Wall
VIRT_RT ?= runtime.h
OBJ = build/workspace.o build/parspl.o build/main.o build/virtual_main.o

.PHONY: clean all

all: clean run

clean:
	rm -rf build

build:
	mkdir -p $@

build/%.o: %.c $(VIRT_RT) | build
	gcc $(CFLAGS) -gdwarf-3 -gstrict-dwarf -c $< -o $@

build/virtual_main: $(OBJ)
	#prerequisites: $?
	#output: $@
	gcc $(CFLAGS) -lpthread $^ -o $@

run: build/virtual_main
	./build/virtual_main
