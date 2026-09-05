# Process Safety Progress 투고 수준 강화 계획

## 1. 목표와 논문 정체성

목표 저널은 *Process Safety Progress* (PSP)이다. 논문의 정체성은 양자 알고리즘
논문이 아니라 다음과 같은 **가스배관 희소사고 위험평가 연구**로 고정한다.

> PHMSA 사고·노출자료로 사고 빈도와 사고 발생 시 중대화 확률을 추정하고,
> 이질적 위험 시나리오의 집계 확률을 quantum amplitude estimation (QAE)으로
> 계산할 때 얻을 수 있는 질의 효율과 현재 회로의 실행 한계를 함께 정량화한다.

잠정 영문 제목:

> **Data-driven rare-event risk quantification for gas pipelines using quantum
> amplitude estimation: Statistical uncertainty, query efficiency, and hardware
> feasibility**

핵심 독자는 양자컴퓨팅 연구자보다 공정안전·배관안전 실무자이다. 따라서 결과의
순서는 `안전 의사결정 결과 → 통계적 신뢰성 → 양자 계산 결과 → 한계`로 둔다.

## 2. 현재 예비연구에서 본문으로 가져갈 것

- PHMSA 사고 시나리오 1,932건과 Serious Incident 49건.
- 2017–2025 운영사-연도 11,444행, 총 3,218,703 mile-years.
- Serious Incident의 PHMSA 정의와 자료 처리·누출변수 제외 규칙.
- QAE 상태준비/Grover 회로의 해석식 검증.
- 동일 oracle-query 예산에서 고전 Monte Carlo와 ideal ML-QAE의 RMSE 비교.
- clean ancilla 합성에 따른 깊이·CX 감소.
- 노이즈 증가에 따른 정확도 붕괴와 실행가능 경계.

기존의 동일·독립 확률 `a_K = 1-(1-p)^K`는 회로 단위시험과 재현성 검증에는
유용하지만, 닫힌형 해가 있어 안전 연구의 주된 문제로는 약하다. 본문 핵심결과가
아니라 검증 절 또는 보충자료로 이동한다.

## 3. 강화된 안전 문제 정의

### 3.1 빈도와 중대화 확률의 분리

운영사 또는 위험계층 `j`, 연도 `t`에 대해 다음 두 성분을 분리한다.

1. 사고 빈도: `lambda_jt = E[N_incident,jt] / exposure_jt`
2. 조건부 중대화 확률: `q_j = P(Serious | incident, scenario class j)`

희소 중대사고의 노출량 기반 강도는 개념적으로 다음과 같다.

`mu_jt = exposure_jt * lambda_jt * q_j`

포트폴리오에서 한 건 이상 중대사고가 발생할 확률은, 조건부 독립 기준모형에서
다음과 같다.

`R = 1 - exp(-sum_jt mu_jt)`

독립성 가정은 기본모형일 뿐이다. 연도 공통 충격 또는 운영사 이질성을 추가한
상관 시나리오를 민감도 분석으로 보고한다.

### 3.2 안전 의사결정 사례

논문에는 최소 세 개의 재현 가능한 의사결정 사례를 둔다.

- 대표적인 1년 운영 포트폴리오의 Serious Incident 위험과 95% 불확실성 구간.
- HCA 비중 또는 주요 사고원인 구성이 변할 때 위험이 얼마나 달라지는지.
- 주어진 허용오차에서 고전 Monte Carlo와 QAE가 요구하는 평가 횟수 및 총
  회로비용이 어떻게 달라지는지.

이는 특정 미국 운영사의 안전등급을 매기기 위한 모델이 아니다. 공개자료에 기반한
방법론적 사례이며, 개별 운영사 순위는 제시하지 않는다.

## 4. 통계 모형

### 4.1 사고 빈도 모형

- 분석단위: 2017–2025 운영사-연도.
- 결과: `incident_count`.
- offset: `log(total_miles)`.
- 비교모형: Poisson, negative binomial, 필요 시 hurdle/zero-inflated 모형.
- 주효과 후보: 연도, HCA mileage fraction 및 자료기간 전체에서 정의가 일관된
  사전 지정 변수만 사용한다.
- 반복측정: 운영사 군집 bootstrap 또는 random intercept로 처리한다.
- 미래예측으로 표현할 때에는 연차보고서 공변량을 1년 lag한다.

모형 선택은 AIC 하나로 끝내지 않고 시간외 검증 deviance, calibration 및
예측구간 coverage를 함께 본다.

### 4.2 사고 발생 시 중대화 모형

- 분석단위: 2010–2025 사고 1,932건.
- 결과: Serious Incident 49건.
- 49건만 양성이므로 고차원 자동선택 모형을 금지한다.
- 주분석: Firth/penalized logistic 또는 약한 정보 사전분포를 둔 Bayesian
  logistic 모형.
- 변수 수와 자유도는 사전에 제한하고, 희소 범주는 안전공학적 의미에 따라 묶는다.
- `apparent_cause`와 사고 시점 압력은 사고 후 시나리오 분석에만 사용하며
  사전 예방예측 변수로 표현하지 않는다.
- 시간검증: 2010–2020 개발, 2021–2022 조정, 2023–2025 최종평가.
- 보고: calibration intercept/slope, Brier score, log loss, PR-AUC와 bootstrap
  신뢰구간. ROC-AUC 단독 보고를 금지한다.

### 4.3 불확실성 전파

빈도모형, 중대화모형 및 위험계층 분포를 운영사·연도 cluster bootstrap으로
재적합한다. 각 bootstrap draw에서 최종 포트폴리오 위험을 다시 계산하여 95%
구간을 만든다. QAE의 샷 오차와 통계모형의 데이터 불확실성은 별도로 보고한 뒤
결합 결과도 제시한다.

## 5. 데이터 기반 QAE 재설계

### 5.1 레지스터

기존처럼 위험단위마다 Bernoulli 큐비트를 하나씩 두는 대신 다음 구조를 주분석으로
사용한다.

- `n`개 scenario/index qubit: `N = 2^n`개의 위험계층 또는 불확실성 시나리오.
- 1개 objective qubit: 시나리오별 중대사고 확률을 진폭으로 부호화.
- 필요한 ancilla: 상태준비·제어회전·Grover 합성에 사용.

상태준비 연산 `A`는 다음을 만족한다.

`A|0> = sum_j sqrt(w_j)|j>(sqrt(1-p_j)|0> + sqrt(p_j)|1>)`

따라서 objective qubit의 1 확률은 `a = sum_j w_j p_j`이다. `w_j`와 `p_j`는
각각 PHMSA 위험계층 분포와 적합된 위험모형에서 나온다.

### 5.2 실험 규모

- 정확검증: 4, 8, 16개 위험계층.
- 주 시뮬레이션: 32–256개 계층 또는 bootstrap 시나리오.
- 확장분석: 더 큰 시나리오 수는 회로를 무조건 실행하지 않고 자원추정과 ideal
  oracle-query 실험을 분리한다.
- 실제 QPU: 4개 계층을 필수 최소실험, 8/16개는 backend 보정자료와 예상비용을
  확인한 뒤 실행한다.

### 5.3 반드시 포함할 비용

- oracle/query count.
- logical depth와 gate count.
- backend transpilation 이후 depth, CX/ECR 계열 2-qubit gate 수.
- 상태준비와 조건부 회전 비용.
- ancilla 수와 총 큐비트 수.
- 가능하면 실제 QPU time과 queue를 제외한 execution metadata.

임의의 `p_j`를 lookup table로 적재하면 상태준비 비용이 `O(N)`이 되어 QAE의
질의복잡도 이점이 사라질 수 있다. 이 결과도 숨기지 않고 **실무 적용의 병목**으로
보고한다. 효율적 parametric state preparation이 가능한 경우와 임의 데이터 적재
경우를 분리한다.

## 6. 비교 기준

고전 비교를 naive Monte Carlo 하나로 제한하지 않는다.

- exact enumeration: 작은 계층 수의 정답.
- analytic/closed-form: 가능한 단순 기준모형.
- crude Monte Carlo.
- stratified Monte Carlo 또는 importance sampling.
- randomized quasi-Monte Carlo: 차원이 허용되는 경우.
- ideal ML-QAE.
- noisy simulation 및 소규모 실제 QPU.

평가지표는 absolute error, relative error, RMSE, bias, 95% interval coverage,
평가 횟수, 총 논리/물리 2-qubit gate 비용이다. 동일 query 비교와 동일 총비용
비교를 혼동하지 않는다.

## 7. 필수 민감도·절제 분석

1. Poisson 대 negative binomial 빈도모형.
2. 중대화 모형의 변수군 및 범주 병합.
3. 운영사/연도 bootstrap 방식.
4. 독립 시나리오 대 공통충격 상관 시나리오.
5. 균일 상태준비 대 경험적 가중 상태준비.
6. lookup 상태준비 비용 포함/제외.
7. no-ancilla 대 one-clean 합성.
8. ideal, calibration-derived noise, 실제 QPU.
9. QAE power schedule와 shot allocation.
10. 허용 상대오차 5%, 10%, 20%에 따른 결론 변화.

## 8. PSP 본문 구성과 분량

PSP의 7,000단어 제한을 고려해 다음을 본문에 둔다.

1. Introduction and process-safety motivation
2. PHMSA data and serious-incident definition
3. Frequency–consequence risk model
4. Data-driven QAE formulation and resource accounting
5. Validation design and classical comparators
6. Safety-risk results
7. Quantum query, circuit, noise, and hardware results
8. Practical implications and limitations
9. Conclusions

세부 회로식, 전체 하이퍼파라미터, 추가 bootstrap 표, QASM, 코드 및 원자료
필드 매핑은 supplementary material과 공개 저장소로 보낸다.

## 9. 그림·표 최소 패키지

### 본문 그림

1. PHMSA 자료선정 및 빈도–중대화–QAE 분석 흐름도.
2. 연도별 사고율과 Serious Incident rate 및 불확실성.
3. 중대화 모형의 calibration/temporal validation.
4. 대표 포트폴리오 위험분포와 위험요인 변화 효과.
5. 동일 query 예산에서 고전 방법과 QAE의 RMSE.
6. 시나리오 수에 따른 상태준비 포함 회로자원.
7. noise/QPU 결과와 유용성 경계.

### 본문 표

1. 데이터셋과 제외기준.
2. 빈도·중대화 모형 성능과 calibration.
3. 대표 안전 의사결정 사례.
4. classical/QAE 정확도 및 계산비용.
5. 주장, 적용범위 및 한계.

## 10. 투고 전 통과조건

아래 항목 중 하나라도 충족하지 못하면 PSP 제출본으로 간주하지 않는다.

- 자료 선택 흐름과 중복·매칭·결측 처리 전부 재현 가능.
- Serious Incident 정의와 분석기간이 PHMSA 원자료에 추적 가능.
- 통계모형이 시간외 검증에서 pooled intercept 기준보다 calibration 또는 proper
  scoring rule 중 하나 이상에서 개선되고, 과적합 증거가 통제됨.
- 최소 한 개의 명확한 배관안전 의사결정 사례가 있음.
- QAE 입력이 임의의 pooled `p`가 아니라 실제 적합된 이질적 위험분포임.
- 정확해가 가능한 작은 문제에서 회로·추정기 구현이 검증됨.
- naive MC 외 최소 한 개의 강한 고전 기준과 비교함.
- 상태준비를 포함한 총 회로비용을 보고함.
- 양자우위·속도우위를 주장하지 않고 적용가능 경계를 정량화함.
- 재현 코드, 환경, seed, QASM 및 결과표가 보충자료로 제공됨.

## 11. 계산 분담과 실행 순서

### Codex가 먼저 수행할 가벼운 작업

1. 데이터 품질표와 분석대상 흐름표 생성.
2. 범주 병합 규칙과 사전 지정 변수 확정.
3. Poisson/NB 및 penalized logistic의 quick pilot.
4. 데이터 기반 4/8/16계층 QAE 입력표 생성.
5. exact truth와 작은 회로 단위시험.
6. 논문용 표·그림 생성 스크립트 작성.

### 사용자가 실행할 긴 계산

1. cluster bootstrap 전체 반복.
2. classical comparator 대규모 반복.
3. QAE full query/shot sweep.
4. calibration-derived noisy simulation.
5. 확인 후 실제 IBM QPU 실행.

긴 계산은 실행 전에 예상 시간·출력경로·재시작 방법이 포함된 명령으로 전달한다.

## 12. 현실적인 완료 판정

현재 결과는 강한 feasibility pilot이지만 PSP 제출본은 아니다. 위 설계에서 통계적
안전모형, 의사결정 사례, 데이터 기반 상태준비, 강한 고전 기준, 전체 비용계산까지
완료되면 PSP와의 주제 적합성과 기술적 완성도가 충분히 설득력 있는 수준이 된다.
실제 QPU 결과는 논문을 강화하지만, 작은 QPU 실행 자체보다 위 다섯 요소가 더
중요하다.
