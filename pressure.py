"""
압력 기반 차단 문제 (MINLP)

지금까지의 문제 (연결성 기반)
    "끊긴다" = 공급원에서 그래프 경로가 없다
    -> 흐름 제약으로 선형화 -> CP-SAT 가 0.02초에 최적 증명 (6번 연속 실패)

바꾼 문제 (압력 기반)
    "공급 못 한다" = 우회로가 있어도 압력강하로 수요처 최소압력을 못 맞춘다
    Weymouth:  f_ij = sgn * C_ij * sqrt(|p_i^2 - p_j^2|)
    압력제곱 변수 pi = p^2 로 두면  f^2 = C (pi_i - pi_j)  <- 2차식
    (De Wolf & Smeers 2000 의 변환. GAMS gastrans.gms 와 동일)

    이제 비볼록 MINLP 가 되어 CP-SAT 를 쓸 수 없다.

이중수준 구조
    상위: 공격자가 k개 배관 제거 -> 공급 불가량 최대화
    하위: 운영자가 남은 망에서 압력·유량 최적화 -> 공급량 최대화
    단일수준 근사: 제거 조합을 고정하고 하위문제를 풀어 평가.
                  상위는 전수/휴리스틱/SCIP.

이 스크립트가 답할 질문
    "SCIP(MINLP) 이 이 문제를 몇 초에 푸는가?"
    빨리 풀리면 7번째 실패. 느리면 양자를 논할 근거.
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
import itertools
import time

import numpy as np
import networkx as nx

from belgian import NODES, ARCS, SUPPLY, DEMAND, N

# 물성 (GAMS gastrans.gms 와 동일)
T_GAS = 281.15      # K
RUG = 0.05          # mm
DEN = 0.616         # 공기 대비 밀도
Z = 0.8             # 압축인자


def pipe_const(d_mm, L_km):
    """C2 계수 (De Wolf & Smeers 식). f^2 = C2 (pi_i - pi_j)"""
    lam = 1.0 / (2 * np.log10(3.7 * d_mm / RUG)) ** 2
    return 96.074830e-15 * d_mm ** 5 / lam / Z / T_GAS / L_km / DEN


C2 = {a[0] - 1: pipe_const(a[3], a[4]) for a in ARCS}
PLO = {k: v[2] for k, v in NODES.items()}
PUP = {k: v[3] for k, v in NODES.items()}
SUP_MAX = {k: v[1] for k, v in NODES.items() if v[1] > 0}
DEM = {k: -v[1] for k, v in NODES.items() if v[1] < 0}
ACTIVE = {a[0] - 1 for a in ARCS if a[5] == 1}


def solve_lower(removed, tl=30, verbose=False):
    """
    하위문제: removed 를 제외한 망에서 공급 가능량 최대화.
    변수 f(유량), pi(압력제곱), served(수요별 공급량)
    Weymouth 를 등식 제약으로. 비볼록.
    반환 (공급불가량, 상태)
    """
    from pyscipopt import Model, quicksum, sqrt as scip_sqrt
    m = Model()
    m.hideOutput(not verbose)
    m.setParam("limits/time", tl)

    pi = {n: m.addVar(lb=PLO[n] ** 2, ub=PUP[n] ** 2, name=f"pi{n}")
          for n in NODES}
    f, sgn = {}, {}
    for a in ARCS:
        i = a[0] - 1
        if i in removed:
            continue
        fmax = 60.0
        f[i] = m.addVar(lb=-fmax, ub=fmax, name=f"f{i}")
        if i in ACTIVE:
            # 압축기: 압력 상승 허용, 유량 단방향
            m.addCons(f[i] >= 0)
        else:
            # Weymouth 등식: f*|f| = C2 (pi_i - pi_j)
            # SCIP 은 f*f*sign 을 직접 못 쓰므로 보조변수로 분해
            fp = m.addVar(lb=0, ub=fmax, name=f"fp{i}")
            fm = m.addVar(lb=0, ub=fmax, name=f"fm{i}")
            b = m.addVar(vtype="B", name=f"b{i}")
            m.addCons(f[i] == fp - fm)
            m.addCons(fp <= fmax * b)
            m.addCons(fm <= fmax * (1 - b))
            m.addCons(fp * fp - fm * fm == C2[i] * (pi[a[1]] - pi[a[2]]))

    sup = {n: m.addVar(lb=0, ub=SUP_MAX[n], name=f"s{n}") for n in SUP_MAX}
    srv = {n: m.addVar(lb=0, ub=DEM[n], name=f"d{n}") for n in DEM}

    for n in NODES:
        out = quicksum(f[a[0] - 1] for a in ARCS
                       if a[1] == n and a[0] - 1 in f)
        inn = quicksum(f[a[0] - 1] for a in ARCS
                       if a[2] == n and a[0] - 1 in f)
        net = (sup[n] if n in sup else 0) - (srv[n] if n in srv else 0)
        m.addCons(out - inn == net)

    m.setObjective(quicksum(srv[n] for n in srv), "maximize")
    m.optimize()
    st = m.getStatus()
    if st in ("optimal", "bestsollimit", "timelimit", "gaplimit"):
        try:
            served = m.getObjVal()
        except Exception:
            return None, st
        return sum(DEM.values()) - served, st
    return None, st


def connectivity_unserved(removed):
    G = nx.Graph()
    G.add_nodes_from(NODES)
    for a in ARCS:
        if a[0] - 1 not in removed:
            G.add_edge(a[1], a[2])
    reach = set()
    for s in SUPPLY:
        if s in G:
            reach |= nx.node_connected_component(G, s)
    return sum(v for n, v in DEM.items() if n not in reach)


if __name__ == "__main__":
    print("=" * 80)
    print(" 압력 기반 차단 문제 (MINLP)")
    print("=" * 80)
    print(f"  노드 {len(NODES)}, 배관 {N}, 압축기 {len(ACTIVE)}")
    print(f"  총수요 {sum(DEM.values()):.3f}, 총공급능력 "
          f"{sum(SUP_MAX.values()):.3f}")
    print(f"  압력 하한 범위 {min(PLO.values()):.0f}~{max(PLO.values()):.0f} bar")

    print(f"\n[검증] 무손상 상태 하위문제")
    t0 = time.time()
    u, st = solve_lower(set(), tl=60)
    print(f"  공급불가 {u if u is None else f'{u:.4f}'}  상태 {st}  "
          f"{time.time()-t0:.1f}s")

    print(f"\n[비교] 연결성 기반 vs 압력 기반  (단독 제거)")
    print(f"  {'배관':>4}{'연결성':>10}{'압력기반':>12}{'상태':>12}{'시간':>8}")
    print("  " + "-" * 48)
    diff = 0
    for i in range(min(N, 8)):
        cu = connectivity_unserved({i})
        t0 = time.time()
        pu, st = solve_lower({i}, tl=30)
        dt = time.time() - t0
        mark = ""
        if pu is not None and abs(pu - cu) > 1e-3:
            diff += 1
            mark = "  <<< 차이"
        print(f"  {i:>4}{cu:>10.3f}"
              f"{(f'{pu:.3f}' if pu is not None else '실패'):>12}"
              f"{st:>12}{dt:>7.1f}s{mark}")
    print(f"\n  연결성과 압력 기반이 다른 경우: {diff}/8")
    print("  -> 차이가 있으면 압력 물리가 실제로 결과를 바꾼다는 뜻")
