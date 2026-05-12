"""test_vars20.py — v9"""
import os
os.environ['XILINX_XRT']='/usr'
os.environ['PATH']=os.environ.get('PATH','')+':/usr/local/share/pynq-venv/bin'
import numpy as np,json,time
from pynq import Overlay,allocate

VAR_INDEX={"a":0,"b":1,"c":2,"d":3,"e":4,"g":5}
MAX_TABLES=6;MAX_FACTORS=3;MAX_TERMS=2;MAX_EVAL_POINTS=4
R_CTRL=0x00;R_T01_LO=0x10;R_T01_HI=0x14;R_T23_LO=0x1C;R_T23_HI=0x20
R_T45_LO=0x28;R_T45_HI=0x2C;R_Q=0x34;R_ROUND_EVALS=0x40;R_BARRETT_M=0x50
R_NUM_PAIRS=0x58;R_TABLE_STRIDE=0x60;R_NUM_TABLES=0x68;R_DEGREE=0x70
R_NUM_TERMS=0x78;R_EXPR_TERMS=0x80;R_CHALLENGE=0xA0;R_MODE=0xA8

def wp(ip,lo,hi,a):ip.write(lo,int(a&0xFFFFFFFF));ip.write(hi,int((a>>32)&0xFFFFFFFF))
def sw(ip,t=120.0):
    ip.write(0,1);t0=time.perf_counter()
    while not(ip.read(0)&2):
        if time.perf_counter()-t0>t:raise TimeoutError
    return time.perf_counter()-t0

def main():
    with open("v20_test_meta.json") as f:meta=json.load(f)
    q=meta["q"];bm=(1<<63)//q;ch=meta["challenges"];exp=meta["expected_evals"]
    ts=meta["table_size"];nr=meta["num_vars"]
    print(f"═══ vars20 v9: ts={ts}, rounds={nr} ═══\n")
    ol=Overlay("sumcheck.bit")
    ip=None
    for n in ol.ip_dict:
        if "sumcheck" in n.lower():ip=getattr(ol,n);print(f"IP: {n}");break
    print("Loading tables...")
    tbl={v:np.load(f"v20_table_{v}.npy") for v in "abcdeg"}
    print("Allocating...")
    b01=allocate(shape=(2*ts,),dtype=np.uint32);b23=allocate(shape=(2*ts,),dtype=np.uint32);b45=allocate(shape=(2*ts,),dtype=np.uint32)
    b01[0:ts]=tbl["a"];b01[ts:2*ts]=tbl["b"];b23[0:ts]=tbl["c"];b23[ts:2*ts]=tbl["d"];b45[0:ts]=tbl["e"];b45[ts:2*ts]=tbl["g"]
    print("Flushing...");b01.flush();b23.flush();b45.flush()
    ef=[2,0,1,0, 1,2,0,0]
    for i,v in enumerate(ef):ip.write(R_EXPR_TERMS+i*4,int(v))
    ip.write(R_Q,int(q));ip.write(R_BARRETT_M,int(bm));ip.write(R_TABLE_STRIDE,ts)
    ip.write(R_NUM_TABLES,MAX_TABLES);ip.write(R_DEGREE,2);ip.write(R_NUM_TERMS,2)
    wp(ip,R_T01_LO,R_T01_HI,b01.device_address);wp(ip,R_T23_LO,R_T23_HI,b23.device_address);wp(ip,R_T45_LO,R_T45_HI,b45.device_address)
    print("Running...\n")
    np_=ts//2;tp=0.0;tf=0.0;ok=True
    for r in range(nr):
        ip.write(R_NUM_PAIRS,np_);ip.write(R_MODE,0);t=sw(ip);tp+=t
        ev=[ip.read(R_ROUND_EVALS+i*4) for i in range(3)]
        ip.write(R_CHALLENGE,int(ch[r]));ip.write(R_MODE,1);t2=sw(ip);tf+=t2
        m=ev==exp[r]
        if not m:ok=False
        print(f"  Round {r:2d}: {'PASS' if m else 'FAIL'}  {ev}  (prove:{t*1000:.3f}ms  fold:{t2*1000:.3f}ms)")
        if not m:print(f"           Expected: {exp[r]}")
        np_//=2
    b01.freebuffer();b23.freebuffer();b45.freebuffer()
    print(f"\n  Prove: {tp*1000:.3f}ms  Fold: {tf*1000:.3f}ms  Total: {(tp+tf)*1000:.3f}ms")
    print(f"  {'ALL PASSED' if ok else 'FAILED'}")

if __name__=="__main__":main()
