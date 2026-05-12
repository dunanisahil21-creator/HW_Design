#ifndef SUMCHECK_ROUND_H
#define SUMCHECK_ROUND_H

#include <stdint.h>
#include "ap_int.h"

#define MAX_TABLES      6
#define MAX_TABLE_SIZE  (1 << 20)
#define MAX_DEGREE      3        // was 5, now a*b*c is highest
#define MAX_EVAL_POINTS 4        // was 6, degree 3 needs 4
#define MAX_TERMS       2        // was 7, a*b+c has 2 terms
#define MAX_FACTORS     3        // was 5, a*b*c has 3 factors
#define TABLES_PER_PORT 2
#define NUM_PORTS       3
#define FOLD_CHUNK_SIZE 256

#define MODE_PROVE 0
#define MODE_FOLD  1

static inline uint32_t mod_add(uint32_t a, uint32_t b, uint32_t q) {
    #pragma HLS INLINE
    uint64_t s = (uint64_t)a + (uint64_t)b;
    return (uint32_t)((s >= q) ? (s - q) : s);
}

static inline uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t q) {
    #pragma HLS INLINE
    return (a >= b) ? (a - b) : (uint32_t)((uint64_t)q - (uint64_t)b + (uint64_t)a);
}

static inline uint32_t mod_mul_barrett(uint32_t a, uint32_t b,
                                        uint32_t q, uint32_t barrett_m) {
    #pragma HLS INLINE
    ap_uint<64> p = (ap_uint<64>)a * (ap_uint<64>)b;
    ap_uint<33> p_hi = (ap_uint<33>)(p >> 31);
    ap_uint<65> qm = (ap_uint<65>)p_hi * (ap_uint<65>)(ap_uint<32>)barrett_m;
    ap_uint<33> q_est = (ap_uint<33>)(qm >> 32);
    ap_uint<65> qeq = (ap_uint<65>)q_est * (ap_uint<65>)(ap_uint<32>)q;
    ap_uint<64> r = p - (ap_uint<64>)qeq;
    uint32_t r32 = (uint32_t)(ap_uint<32>)r;
    if (r >= (ap_uint<64>)q) r32 = r32 - q;
    if (r32 >= q) r32 = r32 - q;
    return r32;
}

static inline uint32_t mle_update(uint32_t z, uint32_t o, uint32_t t,
                                   uint32_t q, uint32_t barrett_m) {
    #pragma HLS INLINE
    uint32_t diff = mod_sub(o, z, q);
    uint32_t prod = mod_mul_barrett(diff, t, q, barrett_m);
    return mod_add(prod, z, q);
}

static uint32_t mle_update_fold(uint32_t z, uint32_t o, uint32_t challenge,
                                 uint32_t q, uint32_t barrett_m) {
    uint32_t diff;
    if (o >= z) diff = o - z;
    else diff = (uint32_t)((uint64_t)q - (uint64_t)z + (uint64_t)o);
    ap_uint<64> p = (ap_uint<64>)diff * (ap_uint<64>)challenge;
    ap_uint<33> p_hi = (ap_uint<33>)(p >> 31);
    ap_uint<65> qm = (ap_uint<65>)p_hi * (ap_uint<65>)(ap_uint<32>)barrett_m;
    ap_uint<33> q_est = (ap_uint<33>)(qm >> 32);
    ap_uint<65> qeq = (ap_uint<65>)q_est * (ap_uint<65>)(ap_uint<32>)q;
    ap_uint<64> r = p - (ap_uint<64>)qeq;
    uint32_t r32 = (uint32_t)(ap_uint<32>)r;
    if (r >= (ap_uint<64>)q) r32 = r32 - q;
    if (r32 >= q) r32 = r32 - q;
    uint64_t s = (uint64_t)r32 + (uint64_t)z;
    return (uint32_t)((s >= q) ? (s - q) : s);
}

void sumcheck_kernel(
    uint32_t *tables_01, uint32_t *tables_23, uint32_t *tables_45,
    uint32_t  round_evals[MAX_EVAL_POINTS],
    uint32_t  q, uint32_t  barrett_m,
    uint32_t  num_pairs, uint32_t  table_stride,
    uint32_t  num_tables, uint32_t  degree,
    uint32_t  expr_terms[MAX_TERMS * (MAX_FACTORS + 1)],
    uint32_t  num_terms, uint32_t  challenge, uint32_t  mode
);

#endif
