#include "sumcheck_round.h"
#include <string.h>

#define CHUNK_SIZE FOLD_CHUNK_SIZE

static void do_prove(
    uint32_t *tables_01, uint32_t *tables_23, uint32_t *tables_45,
    uint32_t  round_evals[MAX_EVAL_POINTS],
    uint32_t q, uint32_t barrett_m,
    uint32_t num_pairs, uint32_t table_stride,
    uint32_t num_tables, uint32_t degree,
    uint32_t local_nfactors[MAX_TERMS],
    uint32_t local_factor_idx[MAX_TERMS][MAX_FACTORS],
    uint32_t num_terms,
    uint32_t buf_01_t0[CHUNK_SIZE*2], uint32_t buf_01_t1[CHUNK_SIZE*2],
    uint32_t buf_23_t0[CHUNK_SIZE*2], uint32_t buf_23_t1[CHUNK_SIZE*2],
    uint32_t buf_45_t0[CHUNK_SIZE*2], uint32_t buf_45_t1[CHUNK_SIZE*2]
) {
    uint32_t acc[MAX_EVAL_POINTS];
    #pragma HLS ARRAY_PARTITION variable=acc complete
    for (int d = 0; d < MAX_EVAL_POINTS; d++) {
        #pragma HLS UNROLL
        acc[d] = 0;
    }

    uint32_t num_chunks = (num_pairs + CHUNK_SIZE - 1) / CHUNK_SIZE;

    PROVE_CHUNK_LOOP:
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        uint32_t cs = chunk * CHUNK_SIZE;
        uint32_t cp = (cs + CHUNK_SIZE <= num_pairs) ? (uint32_t)CHUNK_SIZE : (num_pairs - cs);
        uint32_t rd = cp * 2;

        PRD0T0: for (uint32_t i = 0; i < rd; i++) {
            #pragma HLS PIPELINE II=1
            buf_01_t0[i] = tables_01[0 * table_stride + 2 * cs + i];
        }
        PRD0T1: for (uint32_t i = 0; i < rd; i++) {
            #pragma HLS PIPELINE II=1
            buf_01_t1[i] = tables_01[1 * table_stride + 2 * cs + i];
        }
        PRD1T0: for (uint32_t i = 0; i < rd; i++) {
            #pragma HLS PIPELINE II=1
            buf_23_t0[i] = tables_23[0 * table_stride + 2 * cs + i];
        }
        PRD1T1: for (uint32_t i = 0; i < rd; i++) {
            #pragma HLS PIPELINE II=1
            buf_23_t1[i] = tables_23[1 * table_stride + 2 * cs + i];
        }
        PRD2T0: for (uint32_t i = 0; i < rd; i++) {
            #pragma HLS PIPELINE II=1
            buf_45_t0[i] = tables_45[0 * table_stride + 2 * cs + i];
        }
        PRD2T1: for (uint32_t i = 0; i < rd; i++) {
            #pragma HLS PIPELINE II=1
            buf_45_t1[i] = tables_45[1 * table_stride + 2 * cs + i];
        }

        PROVE_PAIR:
        for (uint32_t p = 0; p < cp; p++) {
            uint32_t zeros[MAX_TABLES], ones[MAX_TABLES];
            #pragma HLS ARRAY_PARTITION variable=zeros complete
            #pragma HLS ARRAY_PARTITION variable=ones complete

            zeros[0] = buf_01_t0[2*p]; ones[0] = buf_01_t0[2*p+1];
            zeros[1] = buf_01_t1[2*p]; ones[1] = buf_01_t1[2*p+1];
            zeros[2] = buf_23_t0[2*p]; ones[2] = buf_23_t0[2*p+1];
            zeros[3] = buf_23_t1[2*p]; ones[3] = buf_23_t1[2*p+1];
            zeros[4] = buf_45_t0[2*p]; ones[4] = buf_45_t0[2*p+1];
            zeros[5] = buf_45_t1[2*p]; ones[5] = buf_45_t1[2*p+1];

            EVAL_LOOP:
            for (int t = 0; t < MAX_EVAL_POINTS; t++) {
                #pragma HLS PIPELINE
                if ((uint32_t)t <= degree) {
                    uint32_t ext[MAX_TABLES];
                    #pragma HLS ARRAY_PARTITION variable=ext complete
                    for (int k = 0; k < MAX_TABLES; k++) {
                        #pragma HLS UNROLL
                        if (t == 0) ext[k] = zeros[k];
                        else if (t == 1) ext[k] = ones[k];
                        else ext[k] = mle_update(zeros[k], ones[k], (uint32_t)t, q, barrett_m);
                    }
                    uint32_t composition = 0;
                    for (int term = 0; term < MAX_TERMS; term++) {
                        #pragma HLS UNROLL
                        if ((uint32_t)term < num_terms) {
                            uint32_t nf = local_nfactors[term];
                            uint32_t product = 1;
                            for (int f = 0; f < MAX_FACTORS; f++) {
                                #pragma HLS UNROLL
                                if ((uint32_t)f < nf) {
                                    uint32_t tidx = local_factor_idx[term][f];
                                    product = mod_mul_barrett(product, ext[tidx], q, barrett_m);
                                }
                            }
                            composition = mod_add(composition, product, q);
                        }
                    }
                    acc[t] = mod_add(acc[t], composition, q);
                }
            }
        }
    }

    for (int d = 0; d < MAX_EVAL_POINTS; d++) {
        #pragma HLS UNROLL
        round_evals[d] = acc[d];
    }
}

static void do_fold(
    uint32_t *tables_01, uint32_t *tables_23, uint32_t *tables_45,
    uint32_t q, uint32_t barrett_m,
    uint32_t num_pairs, uint32_t table_stride, uint32_t challenge,
    uint32_t rd_01[CHUNK_SIZE*2], uint32_t rd_23[CHUNK_SIZE*2], uint32_t rd_45[CHUNK_SIZE*2],
    uint32_t wr_01[CHUNK_SIZE], uint32_t wr_23[CHUNK_SIZE], uint32_t wr_45[CHUNK_SIZE]
) {
    uint32_t num_chunks = (num_pairs + CHUNK_SIZE - 1) / CHUNK_SIZE;

    PASS_LOOP:
    for (int tbl = 0; tbl < TABLES_PER_PORT; tbl++) {
        uint32_t base = tbl * table_stride;
        FOLD_CHUNK:
        for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
            uint32_t cs = chunk * CHUNK_SIZE;
            uint32_t cp = (cs + CHUNK_SIZE <= num_pairs) ? (uint32_t)CHUNK_SIZE : (num_pairs - cs);
            uint32_t rdc = cp * 2;

            FRD01: for (uint32_t i = 0; i < rdc; i++) {
                #pragma HLS PIPELINE II=1
                rd_01[i] = tables_01[base + 2 * cs + i];
            }
            FRD23: for (uint32_t i = 0; i < rdc; i++) {
                #pragma HLS PIPELINE II=1
                rd_23[i] = tables_23[base + 2 * cs + i];
            }
            FRD45: for (uint32_t i = 0; i < rdc; i++) {
                #pragma HLS PIPELINE II=1
                rd_45[i] = tables_45[base + 2 * cs + i];
            }
            FCOMP: for (uint32_t j = 0; j < cp; j++) {
                #pragma HLS PIPELINE
                wr_01[j] = mle_update_fold(rd_01[2*j], rd_01[2*j+1], challenge, q, barrett_m);
                wr_23[j] = mle_update_fold(rd_23[2*j], rd_23[2*j+1], challenge, q, barrett_m);
                wr_45[j] = mle_update_fold(rd_45[2*j], rd_45[2*j+1], challenge, q, barrett_m);
            }
            FWR01: for (uint32_t i = 0; i < cp; i++) {
                #pragma HLS PIPELINE II=1
                tables_01[base + cs + i] = wr_01[i];
            }
            FWR23: for (uint32_t i = 0; i < cp; i++) {
                #pragma HLS PIPELINE II=1
                tables_23[base + cs + i] = wr_23[i];
            }
            FWR45: for (uint32_t i = 0; i < cp; i++) {
                #pragma HLS PIPELINE II=1
                tables_45[base + cs + i] = wr_45[i];
            }
        }
    }
}

void sumcheck_kernel(
    uint32_t *tables_01, uint32_t *tables_23, uint32_t *tables_45,
    uint32_t  round_evals[MAX_EVAL_POINTS],
    uint32_t q, uint32_t barrett_m,
    uint32_t num_pairs, uint32_t table_stride,
    uint32_t num_tables, uint32_t degree,
    uint32_t expr_terms[MAX_TERMS * (MAX_FACTORS + 1)],
    uint32_t num_terms, uint32_t challenge, uint32_t mode
) {
    #pragma HLS INTERFACE m_axi port=tables_01 offset=slave bundle=gmem0 depth=2097152
    #pragma HLS INTERFACE m_axi port=tables_23 offset=slave bundle=gmem1 depth=2097152
    #pragma HLS INTERFACE m_axi port=tables_45 offset=slave bundle=gmem2 depth=2097152
    #pragma HLS INTERFACE s_axilite port=round_evals
    #pragma HLS INTERFACE s_axilite port=q
    #pragma HLS INTERFACE s_axilite port=barrett_m
    #pragma HLS INTERFACE s_axilite port=num_pairs
    #pragma HLS INTERFACE s_axilite port=table_stride
    #pragma HLS INTERFACE s_axilite port=num_tables
    #pragma HLS INTERFACE s_axilite port=degree
    #pragma HLS INTERFACE s_axilite port=expr_terms
    #pragma HLS INTERFACE s_axilite port=num_terms
    #pragma HLS INTERFACE s_axilite port=challenge
    #pragma HLS INTERFACE s_axilite port=mode
    #pragma HLS INTERFACE s_axilite port=return

    uint32_t local_nfactors[MAX_TERMS];
    uint32_t local_factor_idx[MAX_TERMS][MAX_FACTORS];
    #pragma HLS ARRAY_PARTITION variable=local_nfactors complete
    #pragma HLS ARRAY_PARTITION variable=local_factor_idx complete dim=0

    for (int t = 0; t < MAX_TERMS; t++) {
        #pragma HLS UNROLL
        local_nfactors[t] = expr_terms[t * (MAX_FACTORS + 1)];
        for (int f = 0; f < MAX_FACTORS; f++) {
            #pragma HLS UNROLL
            local_factor_idx[t][f] = expr_terms[t * (MAX_FACTORS + 1) + 1 + f];
        }
    }

    uint32_t bram_a[CHUNK_SIZE*2], bram_b[CHUNK_SIZE*2];
    uint32_t bram_c[CHUNK_SIZE*2], bram_d[CHUNK_SIZE*2];
    uint32_t bram_e[CHUNK_SIZE*2], bram_g[CHUNK_SIZE*2];

    if (mode == MODE_PROVE) {
        do_prove(tables_01, tables_23, tables_45, round_evals, q, barrett_m,
                 num_pairs, table_stride, num_tables, degree,
                 local_nfactors, local_factor_idx, num_terms,
                 bram_a, bram_b, bram_c, bram_d, bram_e, bram_g);
    } else {
        do_fold(tables_01, tables_23, tables_45, q, barrett_m,
                num_pairs, table_stride, challenge,
                bram_a, bram_c, bram_e, bram_b, bram_d, bram_g);
    }
}
