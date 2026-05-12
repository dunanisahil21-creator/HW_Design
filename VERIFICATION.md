# Synthesis & Verification Evidence

This document provides detailed synthesis reports, simulation logs, and quantitative analysis
for the sumcheck FPGA accelerator.

## 1. HLS C Simulation (csim) — PASS

**Command:** `csim_design` in Vitis HLS 2023.2

**Test case:** v4_case32_0, expression a\*b + c, q = 3,603,169,181, 4 variables (16 entries), 4 rounds

**Console output:**
```
=== v9 testbench ===

Barrett check: PASSED

Round 0: PASS [862479237, 962438882, 2751652296]
Round 1: PASS [340020908, 1881077529, 2383218570]
Round 2: PASS [1215317213, 2726123316, 509838412]
Round 3: PASS [3454689742, 2183503154, 1875025448]

ALL PASSED (v9)
```

The testbench (`sumcheck_tb.cpp`) compares HLS kernel output against hardcoded golden values
from the ECE-9413 assignment test vectors. Each round runs `sumcheck_kernel` in prove mode
(mode=0), verifies the round evaluations match expected values, then runs fold mode (mode=1)
with the corresponding challenge.

## 2. HLS C Synthesis Report — sumcheck_kernel

**Tool:** Vitis HLS 2023.2  
**Part:** xck26-sfvc784-2LV-c (Kria KV260)  
**Target clock:** 5.00 ns (200 MHz)

### Timing Estimate

| Metric | Value |
|--------|-------|
| Target | 5.00 ns |
| Estimated | 3.65 ns |
| Uncertainty | 1.35 ns |
| Slack | +1.35 ns |

### Performance & Resource Estimates

| Module | BRAM | DSP | FF | LUT |
|--------|------|-----|-----|-----|
| sumcheck_kernel (total) | 26 | 372 | 24,703 | 39,560 |
| └ do_prove | 0 | 336 | 17,702 | 29,349 |
| └ do_fold | 0 | 36 | 2,971 | 5,844 |

### Loop Performance (do_prove)

| Loop | Iteration Latency (cycles) | Interval (II) | Trip Count | Pipelined |
|------|---------------------------|---------------|------------|-----------|
| PROVE_CHUNK_LOOP | — | — | 16,777,215 | no |
| PRD_01_T0 (burst read) | 3 | 1 | — | yes |
| PRD_01_T1 (burst read) | 3 | 1 | — | yes |
| PRD_23_T0 (burst read) | 3 | 1 | — | yes |
| PRD_23_T1 (burst read) | 3 | 1 | — | yes |
| PRD_45_T0 (burst read) | 3 | 1 | — | yes |
| PRD_45_T1 (burst read) | 3 | 1 | — | yes |
| PROVE_PAIR_EVAL_LOOP | 33 | 1 | — | yes |

**Key observation:** EVAL_LOOP achieves II=1 (one evaluation point per clock cycle) with
33-cycle pipeline depth. All burst read loops achieve II=1. The prove computation is fully
pipelined within each chunk.

### Loop Performance (do_fold)

| Loop | Iteration Latency (cycles) | Interval (II) | Trip Count | Pipelined |
|------|---------------------------|---------------|------------|-----------|
| PASS_LOOP_FOLD_CHUNK | — | — | 33,554,430 | no |
| FOLD_RD_01 (burst read) | 3 | 1 | 1,023 | yes |
| FOLD_RD_23 (burst read) | 3 | 1 | 1,023 | yes |
| FOLD_RD_45 (burst read) | 3 | 1 | 1,023 | yes |
| FOLD_COMPUTE | 8 | 1 | — | yes |
| FOLD_WR_01 (burst write) | 3 | 1 | — | yes |
| FOLD_WR_23 (burst write) | 3 | 1 | — | yes |
| FOLD_WR_45 (burst write) | 3 | 1 | — | yes |

**Key observation:** Fold compute achieves II=1 with 8-cycle pipeline depth (non-inlined
Barrett multiplication allows HLS to schedule across multiple cycles). All burst read/write
loops achieve II=1.

### HW Interface Summary

| Interface | Type | Data Width | Offset | Bundle |
|-----------|------|-----------|--------|--------|
| tables_01 | m_axi | 32→32 | slave | gmem0 |
| tables_23 | m_axi | 32→32 | slave | gmem1 |
| tables_45 | m_axi | 32→32 | slave | gmem2 |
| s_axi_control | s_axilite | 32 | — | control |
| return | s_axilite (ap_ctrl_hs) | — | — | control |

## 3. RTL Co-simulation (cosim)

**Command:** `cosim_design -tool xsim -rtl verilog`  
**Result:** PASS

The co-simulation runs the same testbench (`sumcheck_tb.cpp`) against the generated Verilog
RTL in Vivado XSIM. All 4 rounds produce bit-exact matching output against the golden test
vectors, confirming that the synthesized hardware matches the C reference behavior.

## 4. Vivado Implementation Report

**Project:** sumcheck_kria_v8  
**Board:** xilinx.com:kv260_som:part0:1.4  
**Tool:** Vivado 2023.2

### Timing Summary (Post-Implementation)

| Metric | Value |
|--------|-------|
| WNS (Worst Negative Slack) | +4.474 ns |
| TNS (Total Negative Slack) | 0.000 ns |
| WHS (Worst Hold Slack) | +0.009 ns |
| WPWS (Worst Pulse Width Slack) | +3.500 ns |
| Failing Endpoints | 0 |
| Total Endpoints | 53,928 |

**All user specified timing constraints are met.**

### Post-Implementation Resource Utilization

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUT | 39,560 | 117,120 | 33.8% |
| LUTRAM | 1,847 | 57,600 | 3.2% |
| FF | 24,703 | 234,240 | 10.5% |
| BRAM | 26 | 144 | 18.1% |
| DSP | 372 | 1,248 | 29.8% |
| URAM | 0 | 64 | 0.0% |
| IO | 0 | 189 | 0.0% |
| BUFG | 1 | 352 | 0.3% |

### Power Estimate (Vivado)

| Component | Power (W) |
|-----------|-----------|
| PL Static | 0.6 W |
| PL Dynamic | ~2.5 W |
| PS | ~2.0 W |
| Total on-chip | ~5.1 W |

## 5. On-Board FPGA Execution Results

### vars4 (16 entries per table, 4 rounds)

```
═══ v9 self-test ═══
Loading: sumcheck.bit
IP: sumcheck_kernel_0
  Round  0: [862479237, 962438882, 2751652296]  (prove:0.025ms  fold:0.010ms)
  Round  1: [340020908, 1881077529, 2383218570]  (prove:0.012ms  fold:0.010ms)
  Round  2: [1215317213, 2726123316, 509838412]  (prove:0.012ms  fold:0.011ms)
  Round  3: [3454689742, 2183503154, 1875025448]  (prove:0.012ms  fold:0.010ms)
  Prove: 0.061ms  Fold: 0.042ms  Total: 0.102ms
  Round 0: PASS
  Round 1: PASS
  Round 2: PASS
  Round 3: PASS
ALL PASSED
```

### vars16 (65,536 entries per table, 16 rounds)

```
═══ vars16 v9: ts=65536, rounds=16 ═══
IP: sumcheck_kernel_0
  Round  0: PASS  [3265444227, 911181640, 1335845044]  (prove:2.808ms  fold:3.176ms)
  Round  1: PASS  [848182148, 1029484804, 580527164]  (prove:1.416ms  fold:1.594ms)
  Round  2: PASS  [1615473922, 850153943, 1471483162]  (prove:0.712ms  fold:0.801ms)
  Round  3: PASS  [2572655797, 903682872, 3249897988]  (prove:0.357ms  fold:0.406ms)
  Round  4: PASS  [95004500, 2027391163, 526722191]  (prove:0.182ms  fold:0.203ms)
  Round  5: PASS  [813009947, 3528012022, 1481527342]  (prove:0.091ms  fold:0.105ms)
  Round  6: PASS  [896619113, 1331644176, 2106381866]  (prove:0.049ms  fold:0.054ms)
  Round  7: PASS  [2363378056, 609013937, 1394030729]  (prove:0.031ms  fold:0.028ms)
  Round  8: PASS  [2739739183, 1804434127, 298471508]  (prove:0.021ms  fold:0.019ms)
  Round  9: PASS  [1881646251, 2342429921, 2115896565]  (prove:0.012ms  fold:0.019ms)
  Round 10: PASS  [2581289253, 747740620, 1007152718]  (prove:0.012ms  fold:0.010ms)
  Round 11: PASS  [3525162145, 2412592020, 1691270024]  (prove:0.011ms  fold:0.010ms)
  Round 12: PASS  [691519542, 2458956712, 627774937]  (prove:0.011ms  fold:0.010ms)
  Round 13: PASS  [2078732239, 2962724668, 2678068427]  (prove:0.011ms  fold:0.010ms)
  Round 14: PASS  [1551398613, 2310732977, 2980056299]  (prove:0.011ms  fold:0.010ms)
  Round 15: PASS  [1755687082, 1227969895, 468751396]  (prove:0.011ms  fold:0.010ms)
  Prove: 5.746ms  Fold: 6.465ms  Total: 12.211ms
  ALL PASSED
```

### vars20 (1,048,576 entries per table, 20 rounds)

```
═══ vars20 v9: ts=1048576, rounds=20 ═══
IP: sumcheck_kernel_0
  Round  0: PASS  [1058408561, 3555923229, 2981525740]  (prove:45.432ms  fold:50.813ms)
  Round  1: PASS  [1439521163, 2462382017, 2342404824]  (prove:22.551ms  fold:25.428ms)
  Round  2: PASS  [743495856, 1917588838, 1887950685]  (prove:11.246ms  fold:12.690ms)
  Round  3: PASS  [3583560840, 2667316385, 2585235382]  (prove:5.707ms  fold:6.349ms)
  Round  4: PASS  [68328916, 3412076759, 3496984283]  (prove:2.818ms  fold:3.180ms)
  Round  5: PASS  [429025399, 3185151424, 2689342172]  (prove:1.409ms  fold:1.593ms)
  Round  6: PASS  [1430161243, 1831208750, 1025540062]  (prove:0.712ms  fold:0.797ms)
  Round  7: PASS  [897365370, 1538818289, 2148119878]  (prove:0.358ms  fold:0.403ms)
  Round  8: PASS  [3248174400, 1488115422, 679705475]  (prove:0.181ms  fold:0.203ms)
  Round  9: PASS  [2281175043, 3496585034, 250011789]  (prove:0.097ms  fold:0.103ms)
  Round 10: PASS  [2935571416, 1738242242, 892927964]  (prove:0.054ms  fold:0.061ms)
  Round 11: PASS  [1682937473, 2169430563, 1305452665]  (prove:0.029ms  fold:0.036ms)
  Round 12: PASS  [2112862847, 167854182, 2339605356]  (prove:0.020ms  fold:0.019ms)
  Round 13: PASS  [1105237219, 2256697790, 2205585191]  (prove:0.020ms  fold:0.019ms)
  Round 14: PASS  [2159202167, 1070590044, 1321752917]  (prove:0.011ms  fold:0.010ms)
  Round 15: PASS  [2460964272, 628246140, 1442774682]  (prove:0.011ms  fold:0.010ms)
  Round 16: PASS  [2677317864, 114594563, 134184484]  (prove:0.011ms  fold:0.010ms)
  Round 17: PASS  [1642987244, 725572733, 3416654036]  (prove:0.010ms  fold:0.010ms)
  Round 18: PASS  [3510239315, 1229197824, 3207587309]  (prove:0.011ms  fold:0.010ms)
  Round 19: PASS  [919726925, 3055678154, 1795524688]  (prove:0.011ms  fold:0.010ms)
  Prove: 90.696ms  Fold: 101.753ms  Total: 192.450ms
  ALL PASSED
```

### All 4 Base Expressions (vars16)

```
═══ All 4 base expressions, vars16 ═══
  [0] a                   : 7.091 ms prove — ALL 16 ROUNDS PASS
  [1] a*b                 : 7.079 ms prove — ALL 16 ROUNDS PASS
  [2] a*b + c             : 7.088 ms prove — ALL 16 ROUNDS PASS
  [3] a*b*c               : 7.066 ms prove — ALL 16 ROUNDS PASS
  Total fold (shared):      6.260 ms
  ALL EXPRESSIONS PASSED
```

## 6. Quantitative Throughput Analysis

### Prove Throughput

For vars16 round 0 (32,768 pairs, largest round):

| Metric | Value | Derivation |
|--------|-------|------------|
| Pairs processed | 32,768 | 65,536 entries / 2 |
| EVAL_LOOP II | 1 cycle | From HLS synthesis report |
| EVAL_LOOP depth | 33 cycles | From HLS synthesis report |
| Eval points per pair | 4 | degree 3 + 1 (for a\*b\*c) |
| Compute cycles per pair | 4 × 1 = 4 cycles | (II=1, 4 iterations) |
| Chunk size | 256 pairs | FOLD_CHUNK_SIZE constant |
| Burst read per chunk | 6 × 512 = 3,072 cycles | 6 tables × 256 pairs × 2 values, II=1 |
| Compute per chunk | 256 × 4 = 1,024 cycles | 256 pairs × 4 eval points |
| Total per chunk | ~4,096 cycles | read + compute (sequential) |
| Chunks for round 0 | 128 | 32,768 / 256 |
| Total cycles round 0 | ~524,288 | 128 chunks × 4,096 |
| Clock period | 5 ns (200 MHz) | |
| **Expected time** | **~2.6 ms** | 524,288 × 5 ns |
| **Measured time** | **2.8 ms** | From PYNQ timer |
| **Overhead** | **7.7%** | AXI transaction setup, pipeline drain |

The measured time closely matches the theoretical calculation, confirming the pipeline
operates at the expected throughput.

### Fold Throughput

For vars16 round 0 (32,768 pairs):

| Metric | Value | Derivation |
|--------|-------|------------|
| Pairs processed | 32,768 | |
| FOLD_COMPUTE II | 1 cycle | From HLS synthesis report |
| Passes | 2 | 2 tables per port |
| Burst read per chunk | 3 × 512 = 1,536 cycles | 3 ports × 512 values |
| Compute per chunk | 256 cycles | II=1, 256 pairs |
| Burst write per chunk | 3 × 256 = 768 cycles | 3 ports × 256 results |
| Total per chunk | ~2,560 cycles | read + compute + write |
| Chunks per pass | 128 | 32,768 / 256 |
| Total cycles | 2 × 128 × 2,560 = 655,360 | 2 passes |
| **Expected time** | **~3.3 ms** | 655,360 × 5 ns |
| **Measured time** | **3.2 ms** | From PYNQ timer |

### Resource Budget Tradeoff

| Design Choice | LUT Cost | DSP Cost | Throughput Impact |
|---|---|---|---|
| Barrett mod_mul (inlined, prove) | ~4,500 LUT per instance | 56 DSP per instance | Eliminates 33 ns divider path |
| Barrett mod_mul (non-inlined, fold) | ~1,200 LUT (shared) | 12 DSP (shared) | Allows II=1 at 200 MHz via multi-cycle scheduling |
| BRAM prefetch (6 buffers × 2 KB) | ~200 LUT (address logic) | 0 DSP | Eliminates ~100 ns DDR latency per read |
| 3× m_axi ports | ~3,000 LUT (AXI logic) | 0 DSP | 3× read bandwidth vs single port |
| Unified kernel mode mux | ~500 LUT | 0 DSP | Enables resource sharing, saves ~50K LUT vs dual-IP |
| EVAL_LOOP pipeline (prove) | ~24,000 LUT | 336 DSP | II=1 per eval point |
| **Total kernel** | **39,560 LUT (34%)** | **372 DSP (30%)** | Both well within KV260 budget |

The design uses 34% of LUTs and 30% of DSPs, leaving 66% LUT and 70% DSP headroom for
potential future additions (e.g., supporting advanced expressions with degree 5).
