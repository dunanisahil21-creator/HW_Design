# Sumcheck Protocol FPGA Accelerator

Hardware-accelerated zero-knowledge proof prover for the sumcheck protocol, deployed on the **AMD Kria KV260** FPGA platform.

---

## Overview

The sumcheck protocol is a fundamental interactive proof used in zero-knowledge proof systems. The prover must evaluate a multilinear polynomial composition at multiple points across multiple rounds, then fold the underlying tables with a verifier-supplied challenge. This project accelerates both operations (prove + fold) on FPGA fabric.

**Key results:**
- **461× speedup** over baseline FPGA implementation (5,630 ms → 12.2 ms on vars16)
- Verified bit-exact across vars4 (16 entries), vars16 (65K entries), and vars20 (1M entries)
- All 4 base expressions supported: `a`, `a*b`, `a*b + c`, `a*b*c`
- Single unified HLS kernel with mode-based prove/fold switching
- Deployed and tested on real KV260 hardware via PYNQ

For detailed synthesis reports, simulation logs, and quantitative analysis, see [VERIFICATION.md](VERIFICATION.md).

## Target Platform

| Spec | Value |
|------|-------|
| Board | AMD Kria KV260 Vision AI Starter Kit |
| FPGA | Zynq UltraScale+ ZU5EV (xck26-sfvc784-2LV-c) |
| LUTs | 117,120 |
| DSP48E2 | 1,248 |
| Block RAM | ~5 MB (144 × 36 Kb) |
| DDR4 | 4 GB, 64-bit @ 2400 MT/s |
| PL Clock | 200 MHz (5 ns) |
| PS | 4× ARM Cortex-A53 @ 1.3 GHz |
| OS | Ubuntu 22.04 + PYNQ 3.0 |

## Repository Structure

```
├── hls_src/                        # Vitis HLS source files
│   ├── sumcheck_round.h            # Header: constants, Barrett reduction, mle_update
│   ├── sumcheck_round.cpp          # Unified kernel: do_prove + do_fold with mode switch
│   └── sumcheck_tb.cpp             # HLS testbench with golden test vectors
│
├── pynq_host/                      # PYNQ Python driver and test scripts
│   ├── sumcheck_pynq.py            # Driver: overlay loading, register MMIO, prove/fold orchestration
│   ├── test_vars16.py              # vars16 test (65K entries, 16 rounds)
│   ├── test_vars20.py              # vars20 test (1M entries, 20 rounds)
│   └── test_all_expressions.py     # All 7 expressions with shared fold
│
├── test_vectors/                   # Test data files
│   ├── v16_table_{a,b,c,d,e,g}.npy # vars16 MLE tables (uint32)
│   ├── v16_test_meta.json          # vars16 metadata (q, challenges, expected evals)
│   ├── v20_table_{a,b,c,d,e,g}.npy # vars20 MLE tables
│   └── v20_test_meta.json          # vars20 metadata
│
├── docs/                           # Documentation and presentation
│   └── Sumcheck_FPGA_Accelerator.pptx
│
├── benchmark/
│   └── benchmark_cpu.py            # CPU baseline (numpy + pure Python)
│
└── README.md
```

## Architecture

### Unified Kernel Design

A single HLS top function `sumcheck_kernel` handles both operations via a `mode` register:

- **mode = 0 (Prove):** Evaluates the composition polynomial at `degree + 1` points by iterating over all (even, odd) pairs in each MLE table. Uses BRAM chunk prefetch (256-pair bursts from DDR into on-chip BRAM) and Barrett modular multiplication.

- **mode = 1 (Fold):** Applies the verifier's challenge to fold all tables in-place. Uses chunked burst read/compute/write with parallel 3-port processing and non-inlined Barrett for timing closure.

Hardware resources are shared between prove and fold since they never execute simultaneously. This allows the full 117K LUT budget for the prove pipeline.

### AXI Interface

```
┌──────────────────────┐         ┌─────────────────────────────┐
│  sumcheck_kernel_0   │         │   Zynq UltraScale+ PS       │
│                      │         │                             │
│  s_axi_control ──────┼────────►│  M_AXI_HPM0_LPD (32-bit)   │
│                      │         │                             │
│  m_axi_gmem0 (a,b) ──┼────────►│  S_AXI_HP0_FPD (128-bit)   │
│  m_axi_gmem1 (c,d) ──┼────────►│  S_AXI_HP1_FPD (128-bit)   │
│  m_axi_gmem2 (e,g) ──┼────────►│  S_AXI_HP2_FPD (128-bit)   │
│                      │         │                             │
│  interrupt ──────────┼────────►│  pl_ps_irq0                 │
└──────────────────────┘         └─────────────────────────────┘
```

### Interface Protocol Details

**Control Interface (s_axilite):**
- Protocol: AXI4-Lite (32-bit data, 9-bit address)
- Handshake: `ap_ctrl_hs` — PS writes bit 0 of CTRL register (0x00) to assert `ap_start`, polls bit 1 (`ap_done`) to detect completion
- All scalar parameters (q, barrett_m, num_pairs, mode, etc.) are written as 32-bit MMIO writes before asserting `ap_start`
- Output array `round_evals` is read as 4 consecutive 32-bit MMIO reads from offset 0x40 after `ap_done`
- DDR pointers (`tables_01`, `tables_23`, `tables_45`) are written as two 32-bit registers each (low + high) to support 64-bit physical addresses

**Memory Interfaces (m_axi):**
- Protocol: AXI4 Memory-Mapped (full AXI4, not AXI4-Stream)
- Three independent ports: `gmem0`, `gmem1`, `gmem2` — each 32-bit data width, 64-bit address width
- Connected to Zynq PS HP slave ports (S_AXI_HP0/HP1/HP2_FPD) at 128-bit, with automatic width conversion by AXI SmartConnect
- Burst length: up to 16 beats (HLS default), used for chunk prefetch reads and fold write-back
- Address range: 0x00000000–0x7FFFFFFF (2 GB DDR4 low region)

**DMA / Buffer Management:**
- PS allocates physically contiguous buffers using `pynq.allocate()` (backed by CMA — Contiguous Memory Allocator)
- Physical addresses passed to the kernel via the s_axilite pointer registers
- `buf.flush()` ensures ARM D-cache coherency before kernel reads; `buf.invalidate()` before PS reads FPGA-written data

**Interrupt:**
- Single wire from kernel `interrupt` output to Zynq `pl_ps_irq0[0:0]`
- Not used in current driver (polling-based via `ap_done`); available for interrupt-driven operation



### Key Optimizations

| Optimization | Description | Impact |
|---|---|---|
| Barrett Reduction | Replaced `% q` hardware divider with multiply-based Barrett using `ap_uint` types mapped to DSP48E2 slices | 19× prove speedup |
| 3× DDR Ports | Split 6 tables across 3 independent AXI HP ports (128-bit each) for parallel reads | 3× memory bandwidth |
| BRAM Chunk Prefetch | Burst-read 256-pair chunks into on-chip BRAM; process at 1-cycle latency vs ~100 ns DDR | 7× prove speedup |
| Unified Kernel | Single IP with mode register shares resources between prove and fold | Fits on KV260 |
| Stride Addressing | Fixed `table_stride` eliminates ARM buffer reallocation between rounds | 53 ms eliminated |
| FPGA Fold | Table folding on FPGA with chunked burst + parallel 3-port writes | 755× fold speedup |
| Tightened Bounds | Reduced MAX_DEGREE=3, MAX_TERMS=2, MAX_FACTORS=3 for base polynomials | Smaller pipeline |

## Performance Results

### Final Performance (v9)

| Test Case | Table Entries | Rounds | Prove | Fold | Total |
|---|---|---|---|---|---|
| vars4 | 16 | 4 | 0.06 ms | 0.04 ms | 0.10 ms |
| vars16 | 65,536 | 16 | 5.7 ms | 6.5 ms | 12.2 ms |
| vars20 | 1,048,576 | 20 | 90.7 ms | 101.8 ms | 192.4 ms |

### Optimization Journey (vars16)

| Version | Prove | Fold | Total | Speedup |
|---|---|---|---|---|
| v3: Baseline (divider + ARM fold) | 724 ms | 4,906 ms | 5,630 ms | 1× |
| v9: Final (all optimizations) | 5.7 ms | 6.5 ms | 12.2 ms | **461×** |

### Resource Utilization

| Resource | Used | Available | Utilization |
|---|---|---|---|
| LUT | 39,560 | 117,120 | 34% |
| Flip-Flops | 24,703 | 234,240 | 11% |
| DSP48E2 | 372 | 1,248 | 30% |
| Block RAM (36 Kb) | 26 | 144 | 18% |

**Timing:** Target 5.00 ns (200 MHz), Estimated 3.65 ns, Vivado WNS +4.47 ns. All timing constraints met.

## Build Flow

### Prerequisites

- Vitis HLS 2023.2
- Vivado 2023.2 with KV260 board files
- KV260 running Ubuntu 22.04 + PYNQ 3.0

### Step 1: Vitis HLS — Build the IP

```bash
cd hls_src/

# In Vitis HLS GUI:
# 1. Create project, add sumcheck_round.cpp + .h as Sources
# 2. Add sumcheck_tb.cpp as Testbench
# 3. Set top function: sumcheck_kernel
# 4. Set part: xck26-sfvc784-2LV-c, clock: 5 ns
# 5. Run C Simulation (csim) → verify PASS
# 6. Run C Synthesis → check LUT/DSP/timing
# 7. Export RTL → IP Catalog
```

### Step 2: Vivado — Build the Bitstream

```bash
# In Vivado GUI:
# 1. Create project targeting KV260 board
# 2. Add HLS IP to repository (Settings → IP → Repository)
# 3. Create Block Design:
#    - Add Zynq PS → Run Block Automation
#    - Configure PS: HPM0_LPD (32-bit), HP0/HP1/HP2 (128-bit), IRQ0
#    - Add sumcheck_kernel IP
#    - Run Connection Automation (gmem0→HP0, gmem1→HP1, gmem2→HP2)
#    - Wire interrupt → pl_ps_irq0
# 4. Validate Design (F6)
# 5. Create HDL Wrapper → Generate Bitstream
# 6. Copy .bit + .hwh files (must share same base name)
```

### Step 3: Deploy to KV260

```bash
# Copy files to KV260
scp sumcheck.bit sumcheck.hwh sumcheck_pynq.py ubuntu@<kv260-ip>:~/sumcheck/

# Copy test data
scp v16_table_*.npy v16_test_meta.json ubuntu@<kv260-ip>:~/sumcheck/
scp v20_table_*.npy v20_test_meta.json ubuntu@<kv260-ip>:~/sumcheck/

# Run on KV260
ssh ubuntu@<kv260-ip>
cd ~/sumcheck/
sudo /usr/local/share/pynq-venv/bin/python3 sumcheck_pynq.py sumcheck.bit
sudo /usr/local/share/pynq-venv/bin/python3 test_vars16.py
sudo /usr/local/share/pynq-venv/bin/python3 test_vars20.py
```


## Automated Verification & Reproduction

### Full HLS Flow (one command)

```bash
cd hls_src/
vitis_hls -f run_hls.tcl
```

This single command runs the entire HLS flow end-to-end:
1. Creates project with `sumcheck_kernel` as top function
2. Runs **csim** (C simulation against golden test vectors) → prints PASS/FAIL
3. Runs **csynth** (C synthesis targeting xck26 at 200 MHz) → generates timing/resource reports
4. Runs **cosim** (RTL co-simulation via Vivado XSIM) → verifies generated Verilog matches C
5. Exports packaged IP to `sumcheck_hls_proj/solution1/impl/ip/`

No manual GUI interaction required. All results are printed to stdout and saved in `sumcheck_hls_proj/solution1/syn/report/` and `sumcheck_hls_proj/solution1/sim/report/`.

### On-Board FPGA Tests (after bitstream deployment)

```bash
# SSH into KV260
ssh ubuntu@<kv260-ip>
cd ~/sumcheck/

# Self-test: vars4, expression a*b+c, 4 rounds
sudo /usr/local/share/pynq-venv/bin/python3 sumcheck_pynq.py sumcheck.bit

# vars16: 65,536 entries, 16 rounds
sudo /usr/local/share/pynq-venv/bin/python3 test_vars16.py

# vars20: 1,048,576 entries, 20 rounds
sudo /usr/local/share/pynq-venv/bin/python3 test_vars20.py

# All 4 base expressions with shared fold
sudo /usr/local/share/pynq-venv/bin/python3 test_all_expressions.py v16
```

Each script loads the FPGA overlay, writes registers via MMIO, runs prove+fold for all rounds, compares outputs against golden test vectors, and prints per-round PASS/FAIL with timing. Exit code 0 = all pass, 1 = failure.

### CPU Baseline Benchmark

```bash
cd benchmark/
python3 benchmark_cpu.py v16
python3 benchmark_cpu.py v20
```

Runs numpy (vectorized) and pure Python sumcheck on the host CPU for comparison against FPGA results.

## Verification

Correctness verified at three levels:

1. **HLS C Simulation (csim):** Bit-exact match against golden test vectors from the ECE-9413 assignment test suite
2. **RTL Co-simulation (cosim):** Generated Verilog verified against C reference via Vivado XSIM
3. **On-board FPGA execution:** All rounds pass for vars4 (16 entries), vars16 (65K entries), and vars20 (1M entries) across all 4 base expressions

## Expression Encoding

Expressions are encoded as a flat array of term descriptors:

```
Each term: [num_factors, table_idx_0, table_idx_1, ..., padded to MAX_FACTORS+1]
Table indices: a=0, b=1, c=2, d=3, e=4, g=5

Example: a*b + c
  Term 0: [2, 0, 1, 0]  → 2 factors: table 0 (a) × table 1 (b)
  Term 1: [1, 2, 0, 0]  → 1 factor:  table 2 (c)
  degree = 2, num_terms = 2
```

## Barrett Reduction

Standard modular multiplication uses `(a * b) % q`, which HLS synthesizes into a 64-bit hardware divider (~33 ns critical path, high LUT usage). Barrett reduction replaces this with:

```
barrett_m = floor(2^63 / q)    // precomputed on host, passed via register

p = a * b                       // 64-bit product
p_hi = p >> 31                  // upper 33 bits
q_est = (p_hi * barrett_m) >> 32  // approximate quotient
r = p - q_est * q               // remainder estimate
if (r >= q) r -= q              // at most 2 corrections
if (r >= q) r -= q
```

All operations map to DSP48E2 multiply-accumulate slices with deterministic latency, enabling tight HLS pipelining.

## Tools & Versions

| Tool | Version |
|---|---|
| Vitis HLS | 2023.2 |
| Vivado | 2023.2 |
| PYNQ | 3.0 |
| Ubuntu (KV260) | 22.04 |
| Python | 3.10 |
| XRT | 2.13 |


**Authors:** Abdulhaseeb Khan, Sahil Dunani
