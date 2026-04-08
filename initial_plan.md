# Project Team
*Abdulhaseeb Khan, Sahil Dunani*
---
## IP Definition
We propose to design a *custom Vitis IP on the AMD Kria KV260* to accelerate the *sumcheck protocol*, focusing on the prover-side arithmetic operations.
The main operation performed by the IP is the multilinear extension fold:
[
T'(i) = T(2i) + r * (T(2i+1) - T(2i))  mod q
]
where:
* (T) = evaluation table
* (r) = verifier challenge
* (q) = field modulus
The IP will perform:
* modular addition
* modular subtraction
* modular multiplication
* pairwise folding
* accumulation / reduction
These operations are well-suited for hardware acceleration because they are *highly parallel, repetitive, and arithmetic-heavy*, making them ideal for FPGA DSP pipelines and parallel memory access.
Equivalent pseudocode:
python
def fold_round(table, r, q):
    out = []
    for i in range(len(table)//2):
        a = table[2*i]
        b = table[2*i+1]
        out.append((a + r*(b-a)) % q)
    return out
---
## IP Architecture
The design will be split into smaller hardware modules:
### 1. Control Module
Uses *AXI4-Lite* for configuration and status.
This module receives:
* input address
* output address
* table size
* modulus
* challenge
* start / done signals
### 2. Memory Interface
Uses *shared memory (AXI memory-mapped interface)* to read input tables from DDR and write results back.
### 3. Fold Engine
Core arithmetic pipeline that computes:
[
a+r(b-a)\bmod q
]
for each pair of table entries.
### 4. Modular Arithmetic Units
Reusable submodules for:
* add
* subtract
* multiply
* modulo reduction
### 5. Host Interface
The PS (ARM cores on KV260) will:
* load test vectors
* configure IP
* start execution
* read results
* compare against JAX golden model
---
## Interface Choice
We will use:
* *AXI4-Lite* → control and configuration
* *AXI memory-mapped shared memory* → bulk data transfer between PS DDR and PL
This is preferred over AXI streaming initially because the sumcheck tables are stored as large arrays in DDR.
---
## Validation Plan
We will run the same test cases on:
1. *JAX golden model*
2. *KV260 hardware IP*
Then compare:
* correctness
* latency
* speedup
This allows us to benchmark the hardware accelerator against the software reference.
