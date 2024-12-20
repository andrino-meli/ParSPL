# ParSPL
ParSPL is a code generator to **parallelise repeated solutions of a sparse linear systems** on embedded platforms.
In a nutshell it is used to extract parallelism from the SpTRSV kernel in a preprocessing step.
You want it if you can trade-off **low compile-time with high runtime performance**.

![ParSPL processing steps](docs/parspl.jpg)

ParSPL stands for Parallel Sparsity Pattern Levearaging linear system solver.
Specifically we solve a linear system:
```
A x = b
```
for x repeatedly and in parallel. Repeatedly means:
- the matrix ```A``` is constant (static data)
- the right hand side vector ```b``` changes (input)
- the vector ```x``` is the computed solution (output)
Because ```A``` is static we can do a compile time decomposition ```A=LDL^T```.

ParSPL then further exploits the problem specific sparsity structure in the triangular ```L``` matrix to maximize concurrent computation.
The preprocessing steps are visualized in **a)** to **f)**.

For example: running an MPC controller using the OSQP solver results in such a computation.


## research publication
ParSPL was developed in the context of thermal and energy management for high performance computing (HPC) chips.
Think: voltage and frequency scaling on steroids.

ParSPL is very good at extracting concurrency from a problem formulation.
For very large problems we achieved a 7x speedup on an 8-core embedded platform.
When using memory streaming hardware extensions one can utilize the concurrency even further.
We demonstrated a **33x speedup** with ParSPL using SSSRs.

The corresponding published paper can be found at TODO and explains the methodology behind ParSPL.
The embedded platform used is the famous snitch-cluster <https://github.com/pulp-platform/snitch_cluster> from the pulp-platform -- an open-hardware RISV-V 8 core architecture with a small scratchpad memory.


## usage
### python setup
ParSPL is a C-code generator written in python.
A virtual environment and all required dependencies are installed by running
```
> make setup
```
### autocompletion
For optional but usefull python argument autocompletion install and globaly activate argcomplete with:
```
>pip3 install argcomplete
>activate-global-python-argcomplete
```
### seting up the linear system problem
To allow for direct experimentation certain linear systems are prestored. List them with
```
> make list
```
TODO: How to integrate parspl into Your project:

### code generation and verification by emulation
Select one and start with code generation.
```
> ./parspl.py --test __HPC_3x3_H2 --codegen --link
```
This generates C code in the `build/_HPC_3x3_H2` directory.
The `--link` option direclty symlinks the resulting files into the `virtual` directory.
There the generated code can be verified on the development machine by using the gcc compiler and the linux pthread library.
```
virtual> make
```
Of course You are ment to include the generated C files into your embedded software workflow.

Play and experiment with many options of `parspl.py`.


## Citing
We hope You find utility in ParSPL we encourage you to:
- put a start on this repo
- cite us:

## directory structure
```
.
├── Makefile
├── README.md
├── parspl.py                   # main script for code generation
├── data_format.py
├── draw_mat_gui.py             # a script where one can custom generate L-matrix layouts by drawing for testing the effectiveness of pattern recognition
├── general.py
├── venv                        # python virtual environment
├── src                         # directory of linear system files
│   ├── dummy_2level.json
│   ├── dummy_collist.json
│   ├── _HPC_3x3_H2.json
│   ├── _HPC_4x4_H2.json
├── build                       # resulting generated C files
│   └── _HPC_3x3_H2
│       ├── scheduled_data.h
│       ├── workspace.c
│       └── workspace.h
└── virtual                     # virtual execution environment simulating a multicore embedded platform with linux pthread (mainly for functional verification and as a usage example)
    ├── Makefile
    ├── runtime.h               # environment specific functions and parameters, **edit according to the target embedded platform**
    ├── parspl.c                # main parspl C file that implements all the corresponding kernels and the scheduler
    ├── parspl.h                # **the function you want to call is `solve(core_id)`**
    ├── main.c                  # repeated calls to the parspl linear system solver
    ├── virtual_main.c          # a wrapper of parspl.c for emulation with pthread
    ├── verify.h
    ├── types.h
    ├── scheduled_data.h -> ../build/_HPC_3x3_H2/scheduled_data.h
    ├── workspace.c -> ../build/_HPC_3x3_H2/workspace.c
    └── workspace.h -> ../build/_HPC_3x3_H2/workspace.h
```

