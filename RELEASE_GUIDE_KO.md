# QSafety QAE 공개 배포 가이드

이 문서는 GitHub, PyPI, Zenodo DOI를 하나의 재현 가능한 릴리스로 묶는
절차를 설명합니다. 최초 공개 전 `CITATION.cff`와 `pyproject.toml`의 저자
정보가 현재 공개 의도와 일치하는지 확인합니다.

## 1. 공개 전 검사

```powershell
python -m ruff check .
python -m pytest
python -m build
python -m twine check dist/*
```

토큰, 원자료, 생성 결과, 개인정보가 Git 추적 대상에 없는지 확인합니다.
배포 파일은 `dist/qsafety_qae-0.1.0.tar.gz`와 wheel 두 개입니다.

## 2. GitHub 저장소

1. 공개 저장소 `qsafety-qae`를 만들고 기본 브랜치를 `main`으로 둡니다.
2. 로컬 저장소를 push합니다.
3. GitHub Actions의 `tests`가 통과하는지 확인합니다.
4. 아직 GitHub Release를 만들지 않습니다. Zenodo 연결을 먼저 합니다.

## 3. PyPI Trusted Publishing

PyPI 계정의 Publishing 설정에서 다음 pending publisher를 등록합니다.

- PyPI project name: `qsafety-qae`
- GitHub owner: 실제 계정 또는 조직명
- Repository: `qsafety-qae`
- Workflow: `publish.yml`
- Environment: `pypi`

API 토큰을 저장소 secret에 넣지 않습니다. GitHub Release가 발행되면
OpenID Connect로 신원을 확인한 워크플로가 PyPI에 자동 게시합니다.

## 4. Zenodo DOI

1. Zenodo에 GitHub 계정으로 로그인합니다.
2. GitHub 연동 화면에서 `qsafety-qae` 저장소를 활성화합니다.
3. GitHub에서 태그 `v0.1.0`에 대한 Release를 발행합니다.
4. Zenodo가 릴리스를 보관하고 버전 DOI와 개념 DOI를 만든 것을 확인합니다.
5. DOI를 `CITATION.cff`, README, 논문의 코드 가용성 문구에 추가합니다.

Zenodo 연동은 저장소가 공개되어 있어야 안정적으로 동작합니다. DOI 생성과
PyPI 게시가 끝난 버전 번호는 재사용하거나 덮어쓸 수 없으므로, Release
발행 직전에 저자명·라이선스·버전·패키지 내용을 마지막으로 확인합니다.

## 5. 최종 확인

새 가상환경에서 다음 명령이 동작해야 합니다.

```powershell
python -m pip install qsafety-qae
python -c "from qsafety import RiskDistribution; print(RiskDistribution([1, 3], [0.1, 0.3]).probability)"
```

GitHub 저장소, PyPI 프로젝트, Zenodo 기록이 모두 같은 버전 `0.1.0`, 같은
라이선스, 같은 저자 정보를 가리키는지 확인합니다.
