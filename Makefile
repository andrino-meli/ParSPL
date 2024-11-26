CC 		?= gcc
CFLAGS 	?= -lpthread -std=gnu99
VIRT_RT ?= runtime.h

TEST ?= _HPC_3x3_H2

.PHONY: setup list codegen clean

setup: ./venv/bin/python
	. venv/bin/activate && pip3 install \
		argcomplete==3.5.1 \
		networkx==3.4.2 \
		matplotlib==3.5.2 \
		scipy==1.14.1

./venv/bin/python:
	python3 -p python3.12 -m venv ./venv

list:
	# Available linear systems tests:
	@ls src/*.json

codegen:
	# Generating code for $(TEST)
	./parspl.py --test "$(TEST)" --level_schedule --heur_graph_cut --alap --codegen --link --sssr

clean:
	rm -rf build

build:
	mkdir -p "$@"
