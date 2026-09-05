"""
FMQA 전체 실험 배치

python run_fmqa.py --plan main       # 권장, GPU 기준 1~2시간
python run_fmqa.py --plan quick      # 약 10분 (QAOA 제외)
python run_fmqa.py --plan qaoa       # QAOA 만
"""
import argparse, json, os, subprocess, sys, time
from datetime import datetime
LOG="fmqa_log.txt"
def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    with open(LOG,"a",encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n$ {cmd}\n{datetime.now()}\n")
    p=subprocess.run(cmd,shell=True,capture_output=True,text=True,
                     encoding="utf-8",errors="replace")
    o=(p.stdout or "")+(p.stderr or ""); print(o,flush=True)
    with open(LOG,"a",encoding="utf-8") as f: f.write(o)
    return p.returncode

# 세 최적화기를 동일 조건으로 비교해야 공정하다.
# QAOA 는 호출 비용이 크므로 반복 수를 맞추되 repeats 를 줄인다.
PLANS={
 "quick": ["--k 3 4 --opt exact sa --n-init 80 --n-iter 200 --repeats 3"],
 "main":  ["--k 3 4 5 --opt exact --n-init 80 --n-iter 200 --repeats 5",
           "--k 3 4 5 --opt sa    --n-init 80 --n-iter 200 --repeats 5",
           "--k 3 4 5 --opt qaoa  --n-init 80 --n-iter 200 --repeats 3"],
 "qaoa":  ["--k 3 4 5 --opt qaoa --n-init 80 --n-iter 200 --repeats 3"],
}
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--plan",default="main",choices=list(PLANS))
    ap.add_argument("--workers",type=int,default=os.cpu_count() or 8)
    ap.add_argument("--tl",type=float,default=15.0)
    ap.add_argument("--eps",type=float,default=0.10)
    a=ap.parse_args()
    t0=time.time()
    print("="*84); print(f" FMQA 배치  계획={a.plan}  워커={a.workers}")
    print("="*84)
    fails=0
    for i,args in enumerate(PLANS[a.plan],1):
        out=f"fmqa_{a.plan}_{i}.json"
        rc=sh(f'"{sys.executable}" fmqa.py {args} --workers {a.workers} '
              f'--tl {a.tl} --eps {a.eps} --out {out}')
        if rc!=0 or not os.path.exists(out):
            fails+=1
            print(f"  [실패] 종료코드 {rc}, 결과파일 없음", flush=True)
    if fails:
        print(f"\n [경고] {fails}개 작업 실패. fmqa_log.txt 확인")
    print(f"\n 전체 완료 {(time.time()-t0)/60:.1f}분")
    print("="*84)
    TOT={2:276,3:2024,4:10626,5:42504}
    TGT={2:28.3329,3:35.4400,4:40.2400,5:44.1580}
    import glob, numpy as np
    print(f" {'설정':<14}{'평균 평가수':>12}{'전수 대비':>11}"
          f"{'비율':>10}{'편차':>9}")
    print(" "+"-"*56)
    for f in sorted(glob.glob(f"fmqa_{a.plan}_*.json")):
        for key,runs in json.load(open(f)).items():
            k=int(key.split("_")[0][1:])
            ne=[r["n_eval"] for r in runs]; rt=[r["best"]/TGT[k] for r in runs]
            print(f" {key:<14}{np.mean(ne):>12.0f}"
                  f"{np.mean(ne)/TOT[k]*100:>10.1f}%"
                  f"{np.mean(rt):>10.4f}{np.std(rt):>9.4f}")
