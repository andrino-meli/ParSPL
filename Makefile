CC 		?= gcc
CFLAGS 	?= -lpthread -std=gnu99
VIRT_RT ?= runtime.h

TEST ?= _HPC_3x3_H2

.PHONY: clean all

all: codegen

setup:
	pip3 install argcomplete
	argcomplete.autocomplete(parser)

list:
	# Available tests for linear systems:
	ls src/*.json

codegen:
	# Generating code for $(TEST)
	./parspl.py --test $(TEST) --level_thr 3 --alap --codegen --link --sssr

clean:
	rm -rf build

build:
	mkdir -p $@
