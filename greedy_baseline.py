"""고전 휴리스틱 기준선: 탐욕 / 국소탐색"""

import os as _os
# OpenMP 런타임 중복 회피 (conda MKL 과 pyscipopt 충돌)
# 반드시 numpy/scipy/pyscipopt import 이전에 설정해야 한다
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys as _sys
if _sys.stdout.encoding and _sys.stdout.encoding.lower() not in ("utf-8","utf8"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import argparse, itertools, json, time
import numpy as np
from belgian import ARCS, N
from pressure import solve_lower, connectivity_unserved, DEM

def ev(c, tl=20):
    u,_=solve_lower(set(c),tl=tl)
    return u if u is not None else -1.0

def greedy(k, tl=20):
    cur=[]
    for _ in range(k):
        best=(-1,None)
        for i in range(N):
            if i in cur: continue
            u=ev(cur+[i],tl)
            if u>best[0]: best=(u,i)
        cur.append(best[1])
    return best[0], sorted(cur)

def local_search(k, start, tl=20, rounds=50):
    cur=list(start); cv=ev(cur,tl)
    for _ in range(rounds):
        imp=False
        for oi,o in enumerate(list(cur)):
            for i in range(N):
                if i in cur: continue
                t=[x for x in cur if x!=o]+[i]
                v=ev(t,tl)
                if v>cv: cur,cv,imp=t,v,True; break
            if imp: break
        if not imp: break
    return cv, sorted(cur)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--k",type=int,nargs="+",default=[2,3,4])
    ap.add_argument("--tl",type=float,default=20.0)
    ap.add_argument("--out",default="greedy_results.json")
    a=ap.parse_args()
    print("="*76); print(" 고전 휴리스틱 기준선 (압력 기반)"); print("="*76)
    print(f"  {'k':>3}{'탐욕':>11}{'조합':>18}{'국소탐색':>11}{'조합':>18}{'시간':>9}")
    print("  "+"-"*70)
    res={}
    for k in a.k:
        t0=time.time()
        g,gc=greedy(k,a.tl)
        l,lc=local_search(k,gc,a.tl)
        dt=time.time()-t0
        print(f"  {k:>3}{g:>11.4f}{str(gc):>18}{l:>11.4f}{str(lc):>18}{dt:>8.0f}s")
        res[str(k)]=dict(greedy=g,greedy_set=gc,local=l,local_set=lc,wall=dt)
        json.dump(res,open(a.out,"w"),indent=2)
    print(f"\n {a.out} 저장")
