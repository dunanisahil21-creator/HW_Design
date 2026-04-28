# Detailed Plan — Sumcheck Protocol Accelerator

## GitHub Repository
https://github.com/dunanisahil21-creator/HW_Design

## Project Team
Abdulhaseeb Khan, Sahil Dunani

---

## 1. Module Definitions

### 1.1 Control Module
**Function:** Manages configuration and handshaking between PS and PL via AXI4-Lite.

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| clk | input | 1 | Clock signal |
| rst_n | input | 1 | Active-low reset |
| s_axi_awaddr | input | 32 | AXI4-Lite write address |
| s_axi_wdata | input | 32 | AXI4-Lite write data |
| s_axi_araddr | input | 32 | AXI4-Lite read address |
| start | output | 1 | Triggers fold engine |
| done | input | 1 | Fold engine completion signal |
| table_size | output | 32 | Number of elements in table |
| modulus_q | output | 64 | Field modulus q |
| challenge_r | output | 64 | Verifier challenge r |
| input_addr | output | 32 | DDR base address of input table |
| output_addr | output | 32 | DDR base address of output table |

**Timing:** PS writes all config registers before asserting start. Done signal is polled by PS.

---

### 1.2 Memory Interface Module
**Function:** Reads input evaluation table from DDR and writes results back using AXI4 master transactions.

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| clk | input | 1 | Clock |
| rst_n | input | 1 | Reset |
| input_addr | input | 32 | Base address of input table in DDR |
| output_addr | input | 32 | Base address of output table in DDR |
| table_size | input | 32 | Number of elements |
| m_axi_araddr | output | 32 | AXI read address |
| m_axi_rdata | input | 64 | AXI read data |
| m_axi_awaddr | output | 32 | AXI write address |
| m_axi_wdata | output | 64 | AXI write data |
| data_out | output | 64 | Data streamed to fold engine |
| data_in | input | 64 | Result from fold engine |
| valid | output | 1 | Data valid signal |

**Timing:** Bursts pairs of table entries (a, b) to fold engine. Writes results back after fold.

---

### 1.3 Fold Engine
**Function:** Core arithmetic pipeline computing T'(i) = T(2i) + r*(T(2i+1) - T(2i)) mod q

| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| clk | input | 1 | Clock |
| rst_n | input | 1 | Reset |
| a_in | input | 64 | T(2i) — even entry |
| b_in | input | 64 | T(2i+1) — odd entry |
| r_in | input | 64 | Verifier challenge r |
| q_in | input | 64 | Field modulus q |
| valid_in | input | 1 | Input data valid |
| result_out | output | 64 | Folded output value |
| valid_out | output | 1 | Output valid |

**Pipeline stages:**
1. Subtract: diff = b - a
2. Multiply: prod = r * diff
3. Add: sum = a + prod
4. Reduce: out = sum mod q

**Supports both 32-bit and 64-bit modes via template parameter.**

---

### 1.4 Modular Arithmetic Units
**Function:** Reusable submodules instantiated inside the Fold Engine.

#### Modular Adder
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| a, b | input | 64 | Operands |
| q | input | 64 | Modulus |
| result | output | 64 | (a + b) mod q |

#### Modular Subtractor
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| a, b | input | 64 | Operands |
| q | input | 64 | Modulus |
| result | output | 64 | (a - b + q) mod q |

#### Modular Multiplier
| Port | Direction | Width | Description |
|------|-----------|-------|-------------|
| a, b | input | 64 | Operands |
| q | input | 64 | Modulus |
| result | output | 64 | (a * b) mod q |

**Note:** All units are parameterized to support both 32-bit and 64-bit widths.

---

### 1.5 Host Interface (PS Software)
**Function:** ARM Cortex-A53 software on KV260 that orchestrates the accelerator.

Responsibilities:
- Allocate input/output buffers in DDR
- Initialize evaluation table T
- Write config registers over AXI4-Lite
- Assert start, poll done
- Read output buffer
- Compare against JAX/Python golden model

---

## 2. Testbench Definition

### 2.1 Golden Model
We will use **Python/JAX** as the golden model:

```python
def fold_round(table, r, q):
    out = []
    for i in range(len(table) // 2):
        a = table[2*i]
        b = table[2*i + 1]
        out.append((a + r * (b - a)) % q)
    return out
```

### 2.2 Test Scenarios

| Test | Description | Expected Result |
|------|-------------|-----------------|
| Unit: Mod Adder | (a + b) mod q for edge cases | Matches Python |
| Unit: Mod Subtractor | (a - b) mod q including underflow | Matches Python |
| Unit: Mod Multiplier | (a * b) mod q for large values | Matches Python |
| Unit: Fold Engine | Single round fold on small table | Matches Python |
| Integration: 32-bit | Full fold on 16-element table | Matches Python |
| Integration: 64-bit | Full fold on 16-element table | Matches Python |
| Stress: vars16 | 2^16 element table, multiple rounds | Matches JAX output |
| Stress: vars20 | 2^20 element table, multiple rounds | Matches JAX output |

### 2.3 Verification Flow
1. Generate test vectors using Python/JAX
2. Write vectors to DDR via PS
3. Run HLS IP on KV260
4. Read output from DDR
5. Compare bit-exact results with Python golden model
6. Log pass/fail per test case

### 2.4 Performance Metrics
- Latency per fold round (clock cycles)
- Throughput (elements per second)
- Resource utilization (LUT, DSP, BRAM)
- Speedup vs PS-only software execution

---

## 3. Incremental Development Plan

### Milestone 1 — Modular Arithmetic Units
- Implement mod_add, mod_sub, mod_mul in HLS
- Unit test each against Python golden model
- Verify 32-bit and 64-bit modes

### Milestone 2 — Fold Engine
- Implement fold pipeline using arithmetic units
- Test on small 4-element table
- Verify pipeline stages and latency

### Milestone 3 — Memory Interface
- Implement AXI4 master read/write
- Test DDR read/write with simple patterns
- Integrate with Fold Engine

### Milestone 4 — Control Module
- Implement AXI4-Lite register map
- Test start/done handshaking with PS
- Full integration test

### Milestone 5 — Full Integration
- Run complete sumcheck round on KV260
- Compare against JAX golden model
- Measure latency and throughput

### Milestone 6 — Optimization
- Pipeline and parallelize fold engine
- Benchmark vars16 and vars20
- Document speedup vs software

---

## 4. Interface Summary

| Interface | Used Between | Purpose |
|-----------|-------------|---------|
| AXI4-Lite | PS ↔ Control Module | Config registers, start/done |
| AXI4 Memory-Mapped | Memory Interface ↔ DDR | Bulk table read/write |
| Internal signals | Fold Engine ↔ Arithmetic Units | Pipelined datapath |
| Internal signals | Memory Interface ↔ Fold Engine | Streaming pairs (a,b) |
