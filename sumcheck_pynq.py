import os
os.environ['XILINX_XRT']='/usr'
os.environ['PATH']=os.environ.get('PATH','')+':/usr/local/share/pynq-venv/bin'

import numpy as np, time
from pynq import Overlay, allocate

VAR_INDEX={"a":0,"b":1,"c":2,"d":3,"e":4,"g":5}
MAX_TABLES=6; MAX_FACTORS=3; MAX_TERMS=2; MAX_EVAL_POINTS=4

R_CTRL=0x00
R_T01_LO=0x10;R_T01_HI=0x14;R_T23_LO=0x1C;R_T23_HI=0x20;R_T45_LO=0x28;R_T45_HI=0x2C
R_Q=0x34;R_ROUND_EVALS=0x40;R_BARRETT_M=0x50;R_NUM_PAIRS=0x58
R_TABLE_STRIDE=0x60;R_NUM_TABLES=0x68;R_DEGREE=0x70;R_NUM_TERMS=0x78
R_EXPR_TERMS=0x80;R_CHALLENGE=0xA0;R_MODE=0xA8

def encode_expression(expression):
    degree=max(len(t) for t in expression)
    nt=len(expression)
    flat=[0]*(MAX_TERMS*(MAX_FACTORS+1))
    for ti,term in enumerate(expression):
        b=ti*(MAX_FACTORS+1)
        flat[b]=len(term)
        for fi,v in enumerate(term): flat[b+1+fi]=VAR_INDEX[v]
    return flat,degree,nt

def wp(ip,lo,hi,a):ip.write(lo,int(a&0xFFFFFFFF));ip.write(hi,int((a>>32)&0xFFFFFFFF))
def sw(ip,t=120.0):
    ip.write(0,1);t0=time.perf_counter()
    while not(ip.read(0)&2):
        if time.perf_counter()-t0>t:raise TimeoutError
    return time.perf_counter()-t0

class SumcheckAccelerator:
    def __init__(self,path):
        print(f"Loading: {path}")
        self.ol=Overlay(path)
        self.ip=None
        for n in self.ol.ip_dict:
            if "sumcheck" in n.lower():self.ip=getattr(self.ol,n);print(f"IP: {n}");break

    def run_sumcheck(self,tables,expression,challenges,q):
        ts=len(list(tables.values())[0]);nr=len(challenges)
        bm=(1<<63)//q
        ef,deg,nt=encode_expression(expression)
        b01=allocate(shape=(2*ts,),dtype=np.uint32)
        b23=allocate(shape=(2*ts,),dtype=np.uint32)
        b45=allocate(shape=(2*ts,),dtype=np.uint32)
        for v,buf,li in[("a",b01,0),("b",b01,1),("c",b23,0),("d",b23,1),("e",b45,0),("g",b45,1)]:
            s=li*ts
            if v in tables:buf[s:s+ts]=tables[v].astype(np.uint32)
            else:buf[s:s+ts]=0
        b01.flush();b23.flush();b45.flush()
        ip=self.ip
        for i,v in enumerate(ef):ip.write(R_EXPR_TERMS+i*4,int(v))
        ip.write(R_Q,int(q));ip.write(R_BARRETT_M,int(bm));ip.write(R_TABLE_STRIDE,ts)
        ip.write(R_NUM_TABLES,MAX_TABLES);ip.write(R_DEGREE,deg);ip.write(R_NUM_TERMS,nt)
        wp(ip,R_T01_LO,R_T01_HI,b01.device_address)
        wp(ip,R_T23_LO,R_T23_HI,b23.device_address)
        wp(ip,R_T45_LO,R_T45_HI,b45.device_address)
        evs=[];np_=ts//2;tp=0.0;tf=0.0
        for r in range(nr):
            ip.write(R_NUM_PAIRS,np_);ip.write(R_MODE,0);t=sw(ip);tp+=t
            ev=[ip.read(R_ROUND_EVALS+i*4) for i in range(deg+1)];evs.append(ev)
            ip.write(R_CHALLENGE,int(challenges[r]));ip.write(R_MODE,1);t2=sw(ip);tf+=t2
            print(f"  Round {r:2d}: {ev}  (prove:{t*1000:.3f}ms  fold:{t2*1000:.3f}ms)")
            np_//=2
        b01.freebuffer();b23.freebuffer();b45.freebuffer()
        print(f"\n  Prove: {tp*1000:.3f}ms  Fold: {tf*1000:.3f}ms  Total: {(tp+tf)*1000:.3f}ms")
        return evs,tp+tf

def self_test(path):
    Q=3603169181
    ta=np.array([3393379911,2981385125,2084743985,83534990,183830346,1485329884,1196205875,1703421598,3105961542,852416133,697620885,815293416,1462677537,2368463091,1641191852,2695543358],dtype=np.uint32)
    tb=np.array([1009888667,242978280,1489369074,3162735331,3099002876,1015109858,2386803472,882268619,1656912700,1630717979,2285861710,3146833087,900407709,1905237946,1378868269,2702732716],dtype=np.uint32)
    tc=np.array([3026106149,2376905095,1274725382,3008574446,491383016,2313566733,2351566239,3384723696,141235435,2414696771,1018519416,3518918811,3407197093,3201460813,172342382,3401331891],dtype=np.uint32)
    exp=[[862479237,962438882,2751652296],[340020908,1881077529,2383218570],[1215317213,2726123316,509838412],[3454689742,2183503154,1875025448]]
    print("═══ v9 self-test ═══\n")
    sc=SumcheckAccelerator(path)
    ev,_=sc.run_sumcheck({"a":ta,"b":tb,"c":tc},[["a","b"],["c"]],[1569837365,1364485251,2168704920,3350104573],Q)
    ok=all(ev[r]==exp[r] for r in range(4))
    for r in range(4):print(f"  Round {r}: {'PASS' if ev[r]==exp[r] else 'FAIL'}")
    print(f"\n{'ALL PASSED' if ok else 'FAILED'}")
    return ok

if __name__=="__main__":
    import sys;self_test(sys.argv[1] if len(sys.argv)>1 else "sumcheck.bit")
