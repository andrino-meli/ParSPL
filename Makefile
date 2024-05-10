CC 		?= gcc
CFLAGS 	?= -lpthread -std=gnu99
VIRT_RT ?= runtime.h

TEST ?= dummy_fullydense

.PHONY: clean link all

all: link

codegen:
	# Generating code for $(TEST)
	./parspl.py --test $(TEST) --codegen

clean:
	rm -rf build

build:
	mkdir -p $@

link: codegen
	# Linking code of $(TEST) to virtual
	ln -sf ../build/$(TEST)/parspl.c virtual/parspl.c
	ln -sf ../build/$(TEST)/scheduled_data.h virtual/scheduled_data.h
	ln -sf ../build/$(TEST)/workspace.c virtual/workspace.c
	ln -sf ../build/$(TEST)/workspace.h virtual/workspace.h
