import os
os.environ['XILINX_XRT'] = '/usr'
os.environ['PATH'] = os.environ.get('PATH', '') + ':/usr/local/share/pynq-venv/bin'

import numpy as np
import json
import time
from pynq import Overlay, allocate

VAR_INDEX = {"a":0,"b":1,"c":2,"d":3,"e":4,"g":5}
MAX_TABLES = 6
MAX_EVAL_POINTS = 6

R_CTRL=0x00;R_T01_LO=0x10;R_T01_HI=0x14;R_T23_LO=0x1C;R_T23_HI=0x20
R_T45_LO=0x28;R_T45_HI=0x2C;R_Q=0x34;R_ROUND_EVALS=0x40;R_BARRETT_M=0x60
R_NUM_PAIRS=0x68;R_TABLE_STRIDE=0x70;R_NUM_TABLES=0x78;R_DEGREE=0x80
R_NUM_TERMS=0x88;R_CHALLENGE=0x90;R_MODE=0x98;R_EXPR_TERMS=0x100


def encode_expression(expression):
    degree = max(len(term) for term in expression)
    num_terms = len(expression)
    expr_flat = [0] * (7 * MAX_EVAL_POINTS)
    for t_idx, term in enumerate(expression):
        base = t_idx * MAX_EVAL_POINTS
        expr_flat[base] = len(term)
        for f_idx, var_name in enumerate(term):
            expr_flat[base + 1 + f_idx] = VAR_INDEX[var_name]
    return expr_flat, degree, num_terms


def wp(ip, lo, hi, a):
    ip.write(lo, int(a & 0xFFFFFFFF))
    ip.write(hi, int((a >> 32) & 0xFFFFFFFF))


def sw(ip, timeout=120.0):
    ip.write(0x00, 0x01)
    t0 = time.perf_counter()
    while not (ip.read(0x00) & 0x02):
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError("IP timeout")
    return time.perf_counter() - t0


def write_expression(ip, expression):
    expr_flat, degree, num_terms = encode_expression(expression)
    for i, val in enumerate(expr_flat):
        ip.write(R_EXPR_TERMS + i * 4, int(val))
    ip.write(R_DEGREE, degree)
    ip.write(R_NUM_TERMS, num_terms)
    return degree


def run_all_expressions(data_prefix):
    with open(f"{data_prefix}_all_expressions.json") as f:
        meta = json.load(f)

    q = meta["q"]
    barrett_m = (1 << 63) // q
    challenges = meta["challenges"]
    table_size = meta["table_size"]
    num_rounds = meta["num_vars"]
    expressions = meta["expressions"]

    print(f"═══ All 7 expressions, {data_prefix}: q={q}, ts={table_size}, rounds={num_rounds} ═══\n")

    ol = Overlay("sumcheck.bit")
    ip = None
    for n in ol.ip_dict:
        if "sumcheck" in n.lower():
            ip = getattr(ol, n)
            print(f"IP: {n}")
            break

    tbl = {v: np.load(f"{data_prefix}_table_{v}.npy") for v in "abcdeg"}

    buf_01 = allocate(shape=(2 * table_size,), dtype=np.uint32)
    buf_23 = allocate(shape=(2 * table_size,), dtype=np.uint32)
    buf_45 = allocate(shape=(2 * table_size,), dtype=np.uint32)

    buf_01[0:table_size] = tbl["a"]
    buf_01[table_size:2*table_size] = tbl["b"]
    buf_23[0:table_size] = tbl["c"]
    buf_23[table_size:2*table_size] = tbl["d"]
    buf_45[0:table_size] = tbl["e"]
    buf_45[table_size:2*table_size] = tbl["g"]
    buf_01.flush(); buf_23.flush(); buf_45.flush()

    ip.write(R_Q, int(q))
    ip.write(R_BARRETT_M, int(barrett_m))
    ip.write(R_TABLE_STRIDE, table_size)
    ip.write(R_NUM_TABLES, MAX_TABLES)
    wp(ip, R_T01_LO, R_T01_HI, buf_01.device_address)
    wp(ip, R_T23_LO, R_T23_HI, buf_23.device_address)
    wp(ip, R_T45_LO, R_T45_HI, buf_45.device_address)

    expr_prove_times = [0.0] * len(expressions)
    total_fold_time = 0.0
    all_pass = True
    num_pairs = table_size // 2

    for round_idx in range(num_rounds):
        ip.write(R_NUM_PAIRS, num_pairs)

        for expr_info in expressions:
            eidx = expr_info["index"]
            expr = expr_info["expression"]
            degree = write_expression(ip, expr)

            ip.write(R_MODE, 0)  
            t_prove = sw(ip)
            expr_prove_times[eidx] += t_prove

            evals = [ip.read(R_ROUND_EVALS + t * 4) for t in range(degree + 1)]
            expected = expr_info["round_evals"][round_idx]

            if evals != expected:
                expr_str = ' + '.join('*'.join(t) for t in expr)
                print(f"  Round {round_idx:2d} expr[{eidx}] ({expr_str}): FAIL")
                print(f"    Got:      {evals}")
                print(f"    Expected: {expected}")
                all_pass = False

        ip.write(R_CHALLENGE, int(challenges[round_idx]))
        ip.write(R_MODE, 1)  # fold
        t_fold = sw(ip)
        total_fold_time += t_fold

        num_pairs //= 2

        round_prove = sum(expr_prove_times)  
        print(f"  Round {round_idx:2d}: all 7 proved + folded  (fold:{t_fold*1000:.3f}ms)")

    buf_01.freebuffer(); buf_23.freebuffer(); buf_45.freebuffer()

    total_prove = sum(expr_prove_times)

    print(f"\n═══ Results ═══")
    print(f"  Per-expression prove times:")
    for expr_info in expressions:
        eidx = expr_info["index"]
        expr_str = ' + '.join('*'.join(t) for t in expr_info["expression"])
        print(f"    [{eidx}] {expr_str:20s}: {expr_prove_times[eidx]*1000:.3f} ms")

    print(f"\n  Total prove (all 7):  {total_prove*1000:.3f} ms")
    print(f"  Total fold (shared):  {total_fold_time*1000:.3f} ms")
    print(f"  Grand total:          {(total_prove + total_fold_time)*1000:.3f} ms")
    print(f"  Avg per expression:   {(total_prove + total_fold_time)*1000/7:.3f} ms")

    single_expr_time = expr_prove_times[2] + total_fold_time
    print(f"\n  Single expr (a*b+c):  {single_expr_time*1000:.3f} ms")
    print(f"  All 7 expressions:    {(total_prove + total_fold_time)*1000:.3f} ms")
    print(f"  Overhead ratio:       {(total_prove + total_fold_time) / single_expr_time:.2f}x (ideal=1.0x if fold dominated)")

    print(f"\n  {'ALL EXPRESSIONS PASSED' if all_pass else 'SOME EXPRESSIONS FAILED'}")


if __name__ == "__main__":
    import sys
    prefix = sys.argv[1] if len(sys.argv) > 1 else "v16"
    run_all_expressions(prefix)
