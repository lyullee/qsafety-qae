# IBM QPU 논문용 반복실험 규약

## 현재 기준 실행

- 기준일: 2026-09-05 (KST)
- `ibm_fez`: 4096 shots 전체 실행 1회 완료
- `ibm_marrakesh`: 4096 shots 전체 실행 1회 완료
- 각 실행은 4·8·16 위험계층과 Grover power `0, 1, 2`를 포함한다.
- 512-shot `ibm_fez` 실행은 연결·파이프라인 확인용이며 본 반복통계에서 제외한다.
- 누적 QPU 사용량은 27초이다.

## 사전 고정한 실험 설계

1. 실험 단위는 shot이 아니라 **독립 IBM job**이다.
2. 장비별로 서로 다른 보정 시점에서 총 3개 job을 확보한다.
3. 가능하면 실행일을 최소 24시간 간격으로 나눈다.
4. 모든 본실험은 shots=4096, optimization level=3, powers=`0,1,2`로 고정한다.
5. 원시 QPU 결과를 확보할 때까지 오류완화 옵션을 사용하지 않는다.
6. 4·8·16계층을 항상 같은 job에 넣어 동일한 장비 상태에서 비교한다.
7. backend가 비가동 상태이거나 회로 생성에 실패하면 다른 장비로 대체하지 않고 해당
   시점의 실패로 기록한 뒤 다음 시점에 다시 실행한다.

## 평가변수

주평가변수:

- 계층별 `mle_fit_rmse`
- 계층별 `model_mismatch` 발생률
- Grover power별 관측 성공확률과 이상적 성공확률의 차이

부평가변수:

- MLE 절대오차와 상대오차
- ISA depth와 2-qubit gate 수
- backend 및 실행일에 따른 변동

MLE 최종값이 정확하더라도 power-response 궤적이 맞지 않으면 성공으로 판정하지 않는다.
현재 4096-shot 판정 기준은 fit RMSE/max-error 진단에 사용되는 0.05이다.

## 남은 원시 QPU 실행

- 시점 2: 두 장비 각 1회
- 시점 3: 두 장비 각 1회
- 예상 추가 사용량: 약 48초
- 완료 후 장비별 독립 job 수: 3개

각 시점에 먼저 QPU를 사용하지 않는 preview를 실행한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_ibm_qpu_pair.ps1 -Mode preview
```

preview와 사용량을 확인한 뒤에만 실제 쌍 실행을 사용한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_ibm_qpu_pair.ps1 -Mode run -ConfirmQpu
```

한 시점에서 위 `run` 명령은 한 번만 실행한다. 같은 날 반복 클릭하거나 실패 여부를
확인하지 않은 채 다시 실행하지 않는다.

## 집계

QPU를 사용하지 않는 로컬 후처리 명령:

```powershell
C:\Users\lyul\AppData\Local\anaconda3\envs\qae_runtime\python.exe scripts\analyze_ibm_qpu_results.py --shots 4096
```

결과는 `results/qae_heterogeneous/` 아래의 전체 결과, 반복 요약, power-response long table,
PNG/PDF 그림으로 저장된다. 그림의 error bar는 서로 다른 IBM job 사이의 표준편차이다.

## 오류완화 실험

오류완화는 원시 반복실험이 끝난 뒤 별도 실험으로 설계한다. 원시 결과와 오류완화 결과를
한 모집단처럼 합치지 않는다. 우선 4계층을 양성 대조군, 16계층을 실패 대조군으로 사용하고,
추가 QPU 비용을 preview한 뒤 최소 조건만 실행한다.
