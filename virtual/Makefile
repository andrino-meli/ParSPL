CC 		?= gcc
CFLAGS 	?= -lpthread -std=gnu99 -I./
VIRT_RT ?= virt_runtime.h

.PHONY: clean_virt all_virt

clean:
	rm -rf bin

bin:
	mkdir -p $@

bin/sla_%: libsla/%.c $(VIRT_RT) | bin
	$(CC) $(CFLAGS) $< -o $@

link

all: bin/sla_cluster_sdmv
all: bin/sla_cluster_csrmv
