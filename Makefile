CC 		?= gcc
CFLAGS 	?= -lpthread -std=gnu99
VIRT_RT ?= runtime.h

TEST ?= dummy_fullydense

.PHONY: clean link all

all: link

codegen:
	# Generating code for $(TEST)
	./parspl.py --test $(TEST) --codegen --link

clean:
	rm -rf build

build:
	mkdir -p $@
