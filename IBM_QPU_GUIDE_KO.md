# IBM Quantum 무료 QPU 실행 가이드

이 가이드는 PHMSA 4·8·16 위험계층 QAE 회로를 IBM Quantum Open Plan에서
검증하기 위한 절차이다. 실제 QPU 제출 전까지 `check`와 `preview`만 사용한다.

## 0. 비용·보안 원칙

- 사용할 플랜은 반드시 **Open Plan**이다.
- 프로젝트 코드는 `plans_preference=["open"]`, `region="us-east"`로 제한한다.
- API key는 이 대화창, 소스코드, PowerShell 명령줄, 스크린샷에 붙여넣지 않는다.
- 실제 제출은 `--confirm-qpu` 없이는 코드가 차단한다.
- 첫 제출은 4계층·512 shots만 사용한다.
- session mode는 사용하지 않는다.
- 오류완화는 첫 검증에서 끈다.

## 1. IBM Cloud와 IBM Quantum 가입

1. 브라우저에서 <https://quantum.cloud.ibm.com/>을 연다.
2. `Sign in`을 선택한다.
3. IBMid 또는 Google 로그인을 만들거나 사용한다.
4. 안내에 따라 IBM Cloud 계정을 연결 또는 생성한다.
5. IBM Quantum Platform 상단의 account switcher에서 방금 만든 계정을 선택한다.
6. 지역은 `us-east`를 선택한다. Open Plan 인스턴스는 이 지역에서 만든다.

공식 절차: <https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup>

## 2. 무료 Open Plan 인스턴스 생성

1. IBM Quantum Platform에서 `Instances` 페이지로 이동한다.
2. `Create instance`를 선택한다.
3. 플랜에서 `Open Plan`을 선택한다.
4. 지역이 `us-east`인지 확인한다.
5. 알아보기 쉬운 이름, 예를 들어 `qaoa-safety-open`을 입력한다.
6. 생성 후 Instances 목록에서 plan이 `Open`인지 다시 확인한다.
7. 화면에 추가 180분 opt-in 안내가 보이면 조건을 읽고 신청할 수 있다.

Open Plan 기본 한도는 28일 이동 구간당 QPU 10분이다. 인스턴스가 유료
`Pay-As-You-Go`, `Flex`, `Premium`으로 표시되면 중단한다.

## 3. API key 발급

1. IBM Quantum Platform Home/dashboard에서 API key 영역을 찾는다.
2. 새 API key를 생성하거나 표시된 key를 복사한다.
3. 개인 암호관리자 같은 안전한 장소에 보관한다.
4. API key 전체를 타인에게 보내지 않는다.

Qiskit 문서에서는 이 값을 IBM Cloud API key/token으로 부르며, 최신 key는
일반적으로 44자이다. 인스턴스 CRN도 확인할 수 있지만, 현재 프로젝트는
`Open Plan + us-east` 자동선택을 사용하므로 처음에는 입력하지 않아도 된다.

## 4. 로컬에 key를 숨김 입력으로 저장

`qaoa_safety` 프로젝트 PowerShell에서 실행한다.

```powershell
C:\Users\lyul\AppData\Local\anaconda3\envs\qae_runtime\python.exe scripts\ibm_account_setup.py
```

다음 프롬프트가 나타난다.

```text
IBM Quantum API token (input hidden):
```

복사한 key를 붙여넣고 Enter를 누른다. 입력 중 화면에 문자가 보이지 않는 것이
정상이다. 성공하면 다음 메시지가 출력된다.

```text
IBM Quantum Open Plan account saved (region=us-east)
```

## 5. 연결 확인: QPU 시간 0초

```powershell
C:\Users\lyul\AppData\Local\anaconda3\envs\qae_runtime\python.exe scripts\qae_heterogeneous_ibm.py --mode check
```

정상이면 사용 가능한 실제 backend 이름, 큐비트 수와 대기 작업 수가 표시되고
마지막에 다음 문장이 나온다.

```text
No QPU workload was submitted.
```

오류별 확인:

- 인증 오류: API key를 다시 만들고 4단계를 재실행한다.
- instance 없음: Instances 페이지에서 Open Plan과 `us-east`를 확인한다.
- backend 없음: 잠시 후 재실행하거나 IBM 상태 페이지를 확인한다.
- 회사망 TLS/proxy 오류: 개인망에서 시험하거나 사내 보안정책을 확인한다.

## 6. 4계층 smoke preview: QPU 시간 0초

```powershell
C:\Users\lyul\AppData\Local\anaconda3\envs\qae_runtime\python.exe scripts\qae_heterogeneous_ibm.py --mode preview --scenario-qubits 2 --powers 0 1 2 --shots 512
```

이 단계는 backend를 선택하고 실제 장비의 연결구조와 native gate로 transpile하지만
작업을 제출하지 않는다. 다음 파일을 만든다.

- `results/qae_heterogeneous/ibm_preview.csv`
- `results/qae_heterogeneous/ibm_usage_estimate.json`

출력에서 확인할 항목:

- 선택 backend 이름.
- 각 power의 ISA depth.
- native two-qubit gate 수.
- quick-formula 및 duration-based 예상 QPU 초.
- 마지막 문장이 `No QPU workload was submitted.`인지.

이 출력은 실제 제출 전에 검토한다.

## 7. 첫 4계층 QPU smoke test

Preview가 승인된 후에만 다음 명령을 사용한다.

```powershell
C:\Users\lyul\AppData\Local\anaconda3\envs\qae_runtime\python.exe scripts\qae_heterogeneous_ibm.py --mode run --scenario-qubits 2 --powers 0 1 2 --shots 512 --max-execution-time 120 --confirm-qpu
```

제출 직후 job id가 JSON으로 저장된다. 작업 완료 후 다음이 저장된다.

- backend와 job id.
- 실제 good counts.
- ML-QAE 추정치와 정확값 0.0276475의 오차.
- model-mismatch 진단.
- IBM이 반환하는 actual usage.

사용자가 취소하거나 사용자 오류로 실패한 작업은 이미 사용한 QPU 시간이 차감될
수 있으므로 실행 중인 작업을 임의로 취소하지 않는다.

## 8. 전체 preview와 검증

Smoke test가 정상일 때 먼저 전체 preview를 실행한다.

```powershell
C:\Users\lyul\AppData\Local\anaconda3\envs\qae_runtime\python.exe scripts\qae_heterogeneous_ibm.py --mode preview --scenario-qubits 2 3 4 --powers 0 1 2 --shots 4096
```

예상 QPU 시간이 충분히 작고 16계층 ISA 회로가 지나치게 깊지 않으면 전체를
실행한다.

```powershell
C:\Users\lyul\AppData\Local\anaconda3\envs\qae_runtime\python.exe scripts\qae_heterogeneous_ibm.py --mode run --scenario-qubits 2 3 4 --powers 0 1 2 --shots 4096 --max-execution-time 300 --confirm-qpu
```

권장 반복은 첫 전체실행 결과를 본 뒤 3–5회이다. 시뮬레이터의 30회 반복을 QPU에서
그대로 재현하지 않는다. 무료 10분 중 최소 절반은 재실행과 보완시험을 위해 남긴다.

## 9. 결과 해석

- 4계층 성공: 데이터 기반 QAE의 최소 실제장비 검증으로 사용한다.
- 8계층 성능 저하: 회로 깊이 증가에 따른 NISQ 경계로 해석한다.
- 16계층 실패: 숨기지 않고 상태준비·controlled rotation 비용과 hardware
  fidelity의 한계로 보고한다.
- 실제장비 결과를 양자속도우위로 표현하지 않는다.
- queue 대기시간과 QPU usage를 구분한다.

## 10. 실행 순서 요약

1. 가입 및 Open Plan instance 생성.
2. API key 발급.
3. `ibm_account_setup.py`로 숨김 저장.
4. `--mode check` 결과 공유.
5. 4계층 `--mode preview` 결과 공유.
6. 검토 후 512-shot smoke test.
7. 전체 preview.
8. 전체 QPU 검증과 필요한 최소 반복.
