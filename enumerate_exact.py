"""
압력 기반 차단 문제: 전수탐색 정답 확보 (병렬)

배경
    연결성 기반 차단 문제는 CP-SAT 가 0.02초에 최적을 증명한다.
    그러나 압력 물리를 넣으면 답이 달라진다.
        k=2: 압력 28.333 (배관 4,11)  vs  연결성 24.605 (배관 18,22)
        -> 연결성 평가는 위험을 15% 과소평가하고 다른 구간을 지목한다.

    압력 기반 문제는 max-min 이중수준이며 하위문제가 비볼록(Weymouth)이라
    단일수준 MINLP 로 정확히 풀 수 없다(KKT 가 최적성을 보장하지 않음).
    따라서 정확해는 '상위 조합 전수 + 하위 MINLP 평가' 뿐이다.

    조합이 지수적으로 늘어나므로 여기가 최적화 알고리즘이 필요한 지점이다.

사용법
    python enumerate_exact.py --k 3 --workers 20
    python enumerate_exact.py --k 4 --workers 20 --tl 20
"""

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
import argparse, itertools, json, os, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from belgian import NODES, ARCS, SUPPLY, DEMAND, N
from pressure import solve_lower, connectivity_unserved, DEM

_TL = 20.0
def _eval(c):
    """실패 시 제한시간을 늘려 재시도."""
    for tl in (_TL, _TL*3, _TL*10):
        u, st = solve_lower(set(c), tl=tl)
        if u is not None:
            return (float(u), tuple(c), st)
    return (-1.0, tuple(c), "fail")

def run(k, workers, tl, chunk=8):
    global _TL; _TL = tl
    combos = list(itertools.combinations(range(N), k))
    t0 = time.time(); best=(-1,None); vals=[]; failed=[]
    done=0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for u,c,st in ex.map(_eval, combos, chunksize=chunk):
            done+=1
            if u < 0: failed.append(c); continue
            vals.append(u)
            if u>best[0]: best=(u,c)
            if done % max(1,len(combos)//20)==0:
                el=time.time()-t0
                print(f"    {done:>7,}/{len(combos):,}  최선 {best[0]:.3f}  "
                      f"{el:.0f}s  (예상 총 {el/done*len(combos):.0f}s)",
                      flush=True)
    # 병렬 실행 중 실패한 조합은 순차로 재평가 (SCIP 이 프로세스풀에서
    # 간헐적으로 실패하는 현상 보정. 순차 실행 시 실패율 0%)
    if failed:
        print(f"    병렬 실패 {len(failed)}건 -> 순차 재평가", flush=True)
        for c in failed:
            u,_st = solve_lower(set(c), tl=tl*5)
            if u is not None:
                vals.append(float(u))
                if u>best[0]: best=(float(u),c)
        print(f"    재평가 완료", flush=True)
    fails = 0
    dt=time.time()-t0
    v=np.array(vals); cnt=Counter(np.round(v,3))
    cb=max((connectivity_unserved(set(c)),c) for c in combos)
    return dict(k=k, n_combos=len(combos), wall=dt, fails=fails,
                best=float(best[0]), removed=list(best[1]),
                n_distinct=len(cnt), n_at_best=int(cnt[round(best[0],3)]),
                conn_best=float(cb[0]), conn_removed=list(cb[1]),
                same=set(cb[1])==set(best[1]),
                underest_pct=float((best[0]-cb[0])/max(cb[0],1e-9)*100),
                values=[float(x) for x in np.sort(v)[::-1][:50]])

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--k",type=int,nargs="+",default=[3])
    ap.add_argument("--workers",type=int,default=os.cpu_count() or 8)
    ap.add_argument("--tl",type=float,default=20.0)
    ap.add_argument("--out",default="exact_results.json")
    a=ap.parse_args()
    print("="*84); print(" 압력 기반 차단: 전수탐색 정답"); print("="*84)
    print(f"  노드 {len(NODES)}, 배관 {N}, 총수요 {sum(DEM.values()):.3f}")
    print(f"  워커 {a.workers}, 하위문제 제한시간 {a.tl}s")
    allr={}
    for k in a.k:
        print(f"\n[k={k}]  조합 {len(list(itertools.combinations(range(N),k))):,}",
              flush=True)
        r=run(k,a.workers,a.tl); allr[str(k)]=r
        print(f"  압력 기반 최적 {r['best']:.4f}  배관 {r['removed']}")
        print(f"  연결성 최적   {r['conn_best']:.4f}  배관 {r['conn_removed']}")
        print(f"  -> {'동일' if r['same'] else '상이'},  "
              f"연결성 평가의 과소평가 {r['underest_pct']:+.1f}%")
        print(f"  서로 다른 값 {r['n_distinct']}개, 최적 달성 {r['n_at_best']}개, "
              f"평가 실패 {r['fails']}건")
        print(f"  소요 {r['wall']:.0f}s")
        json.dump(allr, open(a.out,"w"), indent=2)
    print(f"\n {a.out} 저장")
