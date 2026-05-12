#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "sumcheck_round.h"

static const uint32_t Q = 3603169181u;
static const uint32_t BM = 2559794329u;
static const uint32_t CH[4] = {1569837365u,1364485251u,2168704920u,3350104573u};
static const uint32_t IA[16] = {3393379911u,2981385125u,2084743985u,83534990u,183830346u,1485329884u,1196205875u,1703421598u,3105961542u,852416133u,697620885u,815293416u,1462677537u,2368463091u,1641191852u,2695543358u};
static const uint32_t IB[16] = {1009888667u,242978280u,1489369074u,3162735331u,3099002876u,1015109858u,2386803472u,882268619u,1656912700u,1630717979u,2285861710u,3146833087u,900407709u,1905237946u,1378868269u,2702732716u};
static const uint32_t IC[16] = {3026106149u,2376905095u,1274725382u,3008574446u,491383016u,2313566733u,2351566239u,3384723696u,141235435u,2414696771u,1018519416u,3518918811u,3407197093u,3201460813u,172342382u,3401331891u};
static const uint32_t EXP[4][3] = {{862479237u,962438882u,2751652296u},{340020908u,1881077529u,2383218570u},{1215317213u,2726123316u,509838412u},{3454689742u,2183503154u,1875025448u}};

int main() {
    printf("=== v9 testbench ===\n\n");
    uint32_t stride=16;
    uint32_t *t01=(uint32_t*)calloc(2*MAX_TABLE_SIZE,4);
    uint32_t *t23=(uint32_t*)calloc(2*MAX_TABLE_SIZE,4);
    uint32_t *t45=(uint32_t*)calloc(2*MAX_TABLE_SIZE,4);
    for(int i=0;i<16;i++){t01[i]=IA[i];t01[stride+i]=IB[i];t23[i]=IC[i];}

    uint32_t expr[MAX_TERMS*(MAX_FACTORS+1)]={0};
    expr[0]=2;expr[1]=0;expr[2]=1; 
    expr[4]=1;expr[5]=2;           

    uint32_t np=8; int err=0;
    for(int r=0;r<4;r++){
        printf("Round %d: ",r);
        uint32_t re[MAX_EVAL_POINTS]={0};
        sumcheck_kernel(t01,t23,t45,re,Q,BM,np,stride,6,2,expr,2,0,MODE_PROVE);
        int ok=1;
        for(int t=0;t<3;t++) if(re[t]!=EXP[r][t]){ok=0;err++;}
        printf("%s [%u,%u,%u]\n",ok?"PASS":"FAIL",re[0],re[1],re[2]);
        sumcheck_kernel(t01,t23,t45,re,Q,BM,np,stride,6,2,expr,2,CH[r],MODE_FOLD);
        np/=2;
    }
    free(t01);free(t23);free(t45);
    printf("\n%s\n",err?"FAILED":"ALL PASSED (v9)");
    return err?1:0;
}
