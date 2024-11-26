# ParSPL
ParSPL is a code generator to **parallelise repeated solutions of a sparse linear systems** on an embedded platform.
In a nutshell it is used to extract parallelism from the SpTRSV kernel in a preprocessing step.
You want it if you can trade-off **low compile-time with high runtime performance**.

ParSPL stands for Parallel Sparsity Pattern Levearaging linear system solver.

Specifically we solve a linear system:
```
A x = b
```
for x repeatedly and in parallel. Repeatedly means:
- the matrix ```A``` is constant (static data)
- the right hand side vector ```b``` changes (input)
- the vector ```x``` is the computed solution (output)

For example: running an MPC controller using the OSQP solver results in such a computation.

# paper
ParSPL was developed in the context of thermal and energy management for high performance computing (HPC) chips.
Think: voltage and frequency scaling on steroids.

ParSPL is very good at extracting concurrency from a problem formulation.
For very large problems we achieved a 7x speedup on an 8-core embedded platform.
When using memory streaming hardware extensions one can utilize the concurrency even further.
We demonstrated a **33x speedup** with ParSPL using SSSRs.

The corresponding published paper is: TODO.
The embedded platform used is the famous snitch-cluster <https://github.com/pulp-platform/snitch_cluster> from the pulp-platform -- an open-hardware RISV-V 8 core architecture with a small scratchpad memory.
Specifically the HPC management was 

# python setup
Create and activate a virtual environment and install required packages
```
python3 -m venv ./venv
source ./venv/bin/activate
pip3 install argcomplete, networkx, matplotlib, scipy, pyqt5, bfloat16
```

In case of some errors consider downgrading some packages.
The following versions are shown to work:
% python3 --version
Python 3.12.4
% pip3 list
Package         Version
--------------- -----------
argcomplete     3.5.1
bfloat16        1.2.0
contourpy       1.3.1
cycler          0.12.1
fonttools       4.55.0
kiwisolver      1.4.7
matplotlib      3.5.3
networkx        3.4.2
numpy           1.26.0
packaging       24.2
pillow          11.0.0
pip             24.0
pyparsing       3.2.0
PyQt5           5.15.11
PyQt5-Qt5       5.15.15
PyQt5_sip       12.15.0
python-dateutil 2.9.0.post0
scipy           1.14.1
six             1.16.0

# Citing
We hope You find utility in ParSPL we encourage you to:
- put a start on this repo
- cite us:
