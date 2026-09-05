"""
FMQA 순환: 2차 대리모형 + 양자 최적화 + 실제 평가

배경
    압력 기반 차단 문제의 목적함수는 MINLP 계산 결과라 QUBO 계수를
    직접 만들 수 없다. FMQA(Kitai et al. 2020, Phys. Rev. Research)는
    블랙박스 함수를 2차 모형으로 학습하고, 그 모형을 양자로 최적화하고,
    제안된 해를 실제 평가해 학습셋에 추가하는 순환이다.

사전 검증 결과 (surrogate_test.py)
    전수 데이터 한 번 학습 시 k=3 에서 R^2=0.960 이나
    대리모형 최적의 실제값은 최적의 63.4% (순위 355/2024).
    -> 회귀가 다수를 차지하는 중간값에 맞춰지고 꼬리를 놓친다.
    -> 순환으로 유망 영역 데이터를 쌓아야 한다. 그 검증이 이 스크립트.

비교 지표
    "몇 번의 MINLP 평가로 최적을 찾는가"
    전수: k=5 에서 42,504회
    탐욕: 수십 회이나 최적의 76.7% 에 그침

최적화기 선택
    --opt exact    대리모형 QUBO 를 전수로 최소화 (기준선, k 고정)
    --opt sa       시뮬레이티드 어닐링
    --opt qaoa     QAOA 상태벡터 (24큐빗, GPU 가능)
"""

import os as _os
# OpenMP 런타임 중복 회피 (conda MKL 과 pyscipopt 충돌)
# 반드시 numpy/scipy/pyscipopt import 이전에 설정해야 한다
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# MINLP 병렬 평가는 프로세스 단위이므로 스레드 1개가 유리.
# 단 QAOA(--opt qaoa) 사용 시 numpy 연산이 많으므로
# 환경변수 OMP_NUM_THREADS 를 미리 지정하면 그 값을 따른다.
_os.environ.setdefault("OMP_NUM_THREADS", "1")


import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from belgian import N
from pressure import solve_lower, connectivity_unserved, DEM

PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]


# ---------------------------------------------------------------------------
def evaluate(c, tl=15):
    u, _ = solve_lower(set(c), tl=tl)
    if u is None:
        u, _ = solve_lower(set(c), tl=tl * 4)
    return float(u) if u is not None else None


def _ev(args):
    c, tl = args
    return (evaluate(c, tl), c)


def features(X):
    return np.hstack([X,
                      np.array([X[:, i] * X[:, j] for i, j in PAIRS]).T,
                      np.ones((len(X), 1))])


def fit(X, y, ridge=1.0):
    F = features(X)
    A = F.T @ F + ridge * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ y)


def to_qubo(w, k, lam=None):
    """
    대리모형 최대화 -> QUBO 최소화.
    w = [선형 N개, 쌍별 |PAIRS|개, 상수]
    제약 sum x = k 는 페널티 (기수 제약).
    """
    Q = np.zeros((N, N))
    for i in range(N):
        Q[i, i] -= w[i]
    for t, (i, j) in enumerate(PAIRS):
        Q[i, j] -= w[N + t]
    scale = max(abs(Q).max(), 1e-9)
    Q /= scale
    lam = lam if lam is not None else 2.0
    for i in range(N):
        Q[i, i] += lam * (1 - 2 * k)
        for j in range(i + 1, N):
            Q[i, j] += 2 * lam
    return Q


# ---------------------------------------------------------------------------
def opt_exact(Q, k, exclude=(), seed=0):
    """대리모형을 전수로 최소화 (k 고정). 24C5=42504 라 즉시."""
    best = (np.inf, None)
    ex = set(map(tuple, exclude))
    for c in itertools.combinations(range(N), k):
        if c in ex:
            continue
        x = np.zeros(N)
        x[list(c)] = 1
        e = float(x @ Q @ x)
        if e < best[0]:
            best = (e, c)
    return best[1]


def opt_sa(Q, k, exclude=(), seed=0, reads=500):
    import neal
    import dimod
    bqm = dimod.BinaryQuadraticModel("BINARY")
    for i in range(N):
        bqm.add_variable(i, float(Q[i, i]))
        for j in range(i + 1, N):
            if abs(Q[i, j]) > 1e-12:
                bqm.add_quadratic(i, j, float(Q[i, j]))
    ss = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=reads,
                                                 seed=seed)
    ex = set(map(tuple, exclude))
    for rec in ss.data(["sample"]):
        c = tuple(sorted(i for i in range(N) if rec.sample[i]))
        if len(c) == k and c not in ex:
            return c
    return opt_exact(Q, k, exclude)


def opt_qaoa(Q, k, exclude=(), seed=0, p=2, restarts=2, shots=4096,
             cpu=False, maxiter=60):
    try:
        from qaoa_gpu import optimize, sample, backend
    except ImportError:
        return opt_sa(Q, k, exclude, seed=seed)
    xp = backend(force_cpu=cpu)
    par, E = optimize(Q, N, p, xp=xp, restarts=restarts, maxiter=maxiter,
                      seed=seed, verbose=False)
    bits = sample(par, E, N, p, shots, xp=xp, seed=seed)
    ex = set(map(tuple, exclude))
    cand = {}
    for row in bits:
        c = tuple(sorted(i for i in range(N) if row[i]))
        if len(c) != k or c in ex:
            continue
        x = np.zeros(N)
        x[list(c)] = 1
        cand[c] = float(x @ Q @ x)
    if not cand:
        return opt_exact(Q, k, exclude)
    return min(cand, key=cand.get)


OPT = {"exact": opt_exact, "sa": opt_sa, "qaoa": opt_qaoa}


# ---------------------------------------------------------------------------
def fmqa(k, n_init=60, n_iter=80, opt="exact", tl=15, workers=8,
         seed=0, verbose=True, target=None, eps=0.10, patience=25):
    """
    eps      : 무작위 탐색 비율. 대리모형 제안 대신 무작위를 뽑을 확률.
               국소최적 정체를 막는다.
    patience : 이 횟수만큼 개선이 없으면 무작위 탐색 비율을 두 배로.
    """
    rng = np.random.default_rng(seed)
    combos = list(itertools.combinations(range(N), k))
    init_idx = rng.choice(len(combos), size=min(n_init, len(combos)),
                          replace=False)
    init = [combos[i] for i in init_idx]

    Xs, ys, seen = [], [], set()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for u, c in ex.map(_ev, [(c, tl) for c in init], chunksize=4):
            if u is None:
                continue
            v = np.zeros(N)
            v[list(c)] = 1
            Xs.append(v)
            ys.append(u)
            seen.add(tuple(c))
    n_eval = len(ys)
    best = max(ys)
    hist = [(n_eval, best, time.time() - t0)]
    if verbose:
        print(f"  초기 {n_eval}회 평가, 최선 {best:.4f} "
              f"({time.time()-t0:.0f}s)")

    stall = 0
    n_random = 0
    for it in range(n_iter):
        # 정체가 길어지면 탐색 비율 상향
        cur_eps = eps * (2.0 if stall >= patience else 1.0)
        if rng.random() < cur_eps:
            # 무작위 탐색 (탐험)
            c = None
            for _ in range(200):
                cand = combos[int(rng.integers(len(combos)))]
                if tuple(cand) not in seen:
                    c = cand
                    n_random += 1
                    break
        else:
            w = fit(np.array(Xs), np.array(ys))
            Q = to_qubo(w, k)
            c = OPT[opt](Q, k, exclude=seen, seed=seed + it)
        if c is None or tuple(c) in seen:
            # 이미 본 해 -> 무작위 탐색으로 대체
            while True:
                c = combos[int(rng.integers(len(combos)))]
                if tuple(c) not in seen:
                    break
        u = evaluate(c, tl)
        if u is None:
            seen.add(tuple(c))
            continue
        v = np.zeros(N)
        v[list(c)] = 1
        Xs.append(v)
        ys.append(u)
        seen.add(tuple(c))
        n_eval += 1
        if u > best + 1e-9:
            best = u
            stall = 0
        else:
            stall += 1
        hist.append((n_eval, best, time.time() - t0))
        if verbose and (it % 10 == 0 or it == n_iter - 1):
            print(f"    반복 {it+1:>3}  평가 {n_eval:>4}  "
                  f"제안 {list(c)} -> {u:.3f}  최선 {best:.4f}")
        if target is not None and best >= target - 1e-6:
            if verbose:
                print(f"    최적 도달! 평가 {n_eval}회 "
                      f"({time.time()-t0:.0f}s)")
            break
    return dict(k=k, opt=opt, n_eval=n_eval, best=best,
                wall=time.time() - t0, history=hist,
                n_random=n_random, eps=eps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--opt", nargs="+", default=["exact"],
                    choices=["exact", "sa", "qaoa"])
    ap.add_argument("--n-init", type=int, default=60)
    ap.add_argument("--n-iter", type=int, default=80)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tl", type=float, default=15.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.10,
                    help="무작위 탐색 비율 (0 이면 기존 동작)")
    ap.add_argument("--patience", type=int, default=25,
                    help="정체 감지 후 탐색 비율 2배")
    ap.add_argument("--out", default="fmqa_results.json")
    a = ap.parse_args()

    TARGET = {2: 28.3329, 3: 35.4400, 4: 40.2400, 5: 44.1580}
    TOTAL = {2: 276, 3: 2024, 4: 10626, 5: 42504}

    print("=" * 84)
    print(" FMQA 순환: 대리모형 + 양자 최적화 + 실제 평가")
    print("=" * 84)
    res = {}
    for k in a.k:
        for o in a.opt:
            runs = []
            for r in range(a.repeats):
                print(f"\n[k={k}, 최적화기={o}, 시행 {r+1}/{a.repeats}]")
                out = fmqa(k, a.n_init, a.n_iter, o, a.tl, a.workers,
                           seed=r, target=TARGET.get(k),
                           eps=a.eps, patience=a.patience)
                runs.append(out)
                print(f"  결과: 평가 {out['n_eval']}회"
                      f"(무작위 {out['n_random']}회), 최선 {out['best']:.4f}"
                      f" / 최적 {TARGET.get(k,'?')}"
                      f"  비율 {out['best']/TARGET[k]:.4f}"
                      f"  ({out['wall']:.0f}s)")
            res[f"k{k}_{o}"] = runs
            ne = [x["n_eval"] for x in runs]
            rt = [x["best"] / TARGET[k] for x in runs]
            print(f"\n  >> k={k} {o}: 평가 {np.mean(ne):.0f}회 "
                  f"(전수 {TOTAL[k]:,}회의 {np.mean(ne)/TOTAL[k]*100:.1f}%), "
                  f"비율 {np.mean(rt):.4f} +- {np.std(rt):.4f}")
            json.dump(res, open(a.out, "w"), indent=2, default=str)
    print(f"\n {a.out} 저장")
