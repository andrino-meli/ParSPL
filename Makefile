CC 		?= gcc
CFLAGS 	?= -lpthread -std=gnu99
VIRT_RT ?= runtime.h

TEST ?= _HPC_3x3_H2

.PHONY: clean link all

all: link

setup:
	pip3 install argcomplete
	argcomplete.autocomplete(parser)

list:
	# Available tests for linear systems:
	ls src/*.json

codegen:
	# Generating code for $(TEST)
	./parspl.py --test $(TEST) --codegen --link

clean:
	rm -rf build

build:
	mkdir -p $@
