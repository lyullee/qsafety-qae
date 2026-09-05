"""
벨기에 가스 수송망 (De Wolf & Smeers 2000)
출처: GAMS Model Library gastrans.gms (SEQ=217)
      D. De Wolf, Y. Smeers, "The Gas Transmission Problem Solved by an
      Extension of the Simplex Algorithm", Management Science 46(11), 2000.
20 노드, 24 배관, 3 압축기(active arc). 실제 벨기에 고열량 가스망.
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
import numpy as np, networkx as nx

# (도시, 공급하한, 공급상한, 압력하한, 압력상한, 비용)
# 음수 = 수요.  단위 백만 m3/일
NODES = {
 "Anderlues":(0,1.2,0,66.2,0),      "Antwerpen":(-1e9,-4.034,30,80.0,0),
 "Arlon":(-1e9,-0.222,0,66.2,0),    "Berneau":(0,0,0,66.2,0),
 "Blaregnies":(-1e9,-15.616,50,66.2,0), "Brugge":(-1e9,-3.918,30,80.0,0),
 "Dudzele":(0,8.4,0,77.0,2.28),     "Gent":(-1e9,-5.256,30,80,0),
 "Liege":(-1e9,-6.385,30,66.2,0),   "Loenhout":(0,4.8,0,77.0,2.28),
 "Mons":(-1e9,-6.848,0,66.2,0),     "Namur":(-1e9,-2.120,0,66.2,0),
 "Petange":(-1e9,-1.919,25,66.2,0), "Peronnes":(0,0.96,0,66.2,1.68),
 "Sinsin":(0,0,0,63.0,0),           "Voeren":(20.344,22.012,50,66.2,1.68),
 "Wanze":(0,0,0,66.2,0),            "Warnand":(0,0,0,66.2,0),
 "Zeebrugge":(8.870,11.594,0,77.0,2.28), "Zomergem":(0,0,0,80.0,0),
}
# (id, from, to, 관경 mm, 길이 km, active)
ARCS = [
 (1,"Zeebrugge","Dudzele",890.0,4.0,0), (2,"Zeebrugge","Dudzele",890.0,4.0,0),
 (3,"Dudzele","Brugge",890.0,6.0,0),    (4,"Dudzele","Brugge",890.0,6.0,0),
 (5,"Brugge","Zomergem",890.0,26.0,0),  (6,"Loenhout","Antwerpen",590.1,43.0,0),
 (7,"Antwerpen","Gent",590.1,29.0,0),   (8,"Gent","Zomergem",590.1,19.0,0),
 (9,"Zomergem","Peronnes",890.0,55.0,0),(10,"Voeren","Berneau",890.0,5.0,1),
 (11,"Voeren","Berneau",395.0,5.0,1),   (12,"Berneau","Liege",890.0,20.0,0),
 (13,"Berneau","Liege",395.0,20.0,0),   (14,"Liege","Warnand",890.0,25.0,0),
 (15,"Liege","Warnand",395.0,25.0,0),   (16,"Warnand","Namur",890.0,42.0,0),
 (17,"Namur","Anderlues",890.0,40.0,0), (18,"Anderlues","Peronnes",890.0,5.0,0),
 (19,"Peronnes","Mons",890.0,10.0,0),   (20,"Mons","Blaregnies",890.0,25.0,0),
 (21,"Warnand","Wanze",395.5,10.5,0),   (22,"Wanze","Sinsin",315.5,26.0,1),
 (23,"Sinsin","Arlon",315.5,98.0,0),    (24,"Arlon","Petange",315.5,6.0,0),
]
SUPPLY={k:v[1] for k,v in NODES.items() if v[1]>0}
DEMAND={k:-v[1] for k,v in NODES.items() if v[1]<0}
N=len(ARCS)

def unserved(removed):
    G=nx.Graph(); G.add_nodes_from(NODES)
    for a in ARCS:
        if a[0]-1 not in removed: G.add_edge(a[1],a[2])
    reach=set()
    for s in SUPPLY:
        if s in G: reach|=nx.node_connected_component(G,s)
    return sum(v for n,v in DEMAND.items() if n not in reach)

if __name__=="__main__":
    from collections import Counter
    import itertools
    dv=np.array(list(DEMAND.values()))
    print("="*72); print(" 벨기에 망 (De Wolf & Smeers 2000)"); print("="*72)
    print(f"  노드 {len(NODES)}, 배관 {N}, 공급 {len(SUPPLY)}, 수요 {len(DEMAND)}")
    print(f"  공급: " + ", ".join(f"{k} {v}" for k,v in SUPPLY.items()))
    print(f"  수요 총 {dv.sum():.3f}, 서로 다른 값 {len(set(dv))}/{len(dv)}")
    print(f"    최소 {dv.min():.3f} 최대 {dv.max():.3f} 변동계수 {dv.std()/dv.mean():.3f}")
    G=nx.Graph(); G.add_nodes_from(NODES)
    for a in ARCS: G.add_edge(a[1],a[2])
    print(f"  연결성분 {nx.number_connected_components(G)}, "
          f"무손상 손실 {unserved(set()):.3f}")
    print(f"\n  단독 제거 손실>0 배관 수: "
          f"{sum(1 for i in range(N) if unserved({i})>0)}/{N}")
    print("\n"+"="*72); print(" 축퇴 검사"); print("="*72)
    for k in (2,3,4):
        vals=[unserved(set(c)) for c in itertools.combinations(range(N),k)]
        c=Counter(np.round(vals,4)); best=max(vals)
        print(f"  k={k}: 조합 {len(vals):>6,}  서로다른값 {len(c):>3}개  "
              f"최적 {best:.3f} 달성 {c[round(best,4)]}개 "
              f"({c[round(best,4)]/len(vals)*100:.3f}%)")
