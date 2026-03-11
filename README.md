# L3MVN 실행 가이드 (iGibson 포팅 기준)

이 문서는 현재 저장소 기준으로 **실행에 필요한 내용만** 정리합니다.

## 1. Docker Setup

### 1-1. 사전 조건
- NVIDIA GPU + 드라이버
- Docker / Docker Compose
- NVIDIA Container Toolkit
- 호스트 경로 접근 가능
  - `/mount/nas2`
  - `/mount/nas3`

### 1-2. Docker 관련 파일
- `docker-compose.yml`: 개발 컨테이너 실행 설정
- `docker/Dockerfile`: CUDA/PyTorch/iGibson 기반 이미지 정의
- `docker/.dockerignore`: 빌드 컨텍스트 제외 파일 설정

### 1-3. 컨테이너 빌드/실행
프로젝트 루트에서 실행:

```bash
docker compose build
docker compose up -d
docker compose exec l3mvn bash
```

종료:

```bash
docker compose down
```

참고:
- 컨테이너 작업 디렉토리: `/mount/nas2/users/dukim/vla_ws/L3MVN`
- iGibson 사용자 설정은 Docker volume `igibson_config`에 저장됨

### 1-4. 이미지 기본 스펙
- Base image: `nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04`
- Python: `3.8`
- PyTorch: `1.7.0` (CUDA 11.0 wheel)
- iGibson: `StanfordVL/iGibson` editable 설치

---

## 2. iGibson Dataset 세팅

현재 코드(`envs/__init__.py`)는 기본 경로를 아래처럼 사용합니다.

- `igibson.assets_path`: `/mount/nas2/users/dukim/vla_ws/igibson/data/assets`
- `igibson.ig_dataset_path`: `/mount/nas2/users/dukim/vla_ws/igibson/data/ig_dataset`
- `igibson.key_path`: `/mount/nas2/users/dukim/vla_ws/igibson/data/igibson.key`

즉, 실행 전에 최소 아래 구조가 있어야 합니다.

```text
/mount/nas2/users/dukim/vla_ws/igibson/data/
  assets/
  ig_dataset/
  igibson.key
```

### 2-1. iGibson 설치 확인
컨테이너 내부에서:

```bash
python -c "import igibson; print(igibson.__file__)"
```

### 2-2. Assets 다운로드
공식 문서 기준 명령:

```bash
python -m igibson.utils.assets_utils --download_assets
```

### 2-3. iGibson Scene Dataset(ig_dataset) 준비
공식 가이드(라이선스 폼 제출 후 key 발급)로 진행합니다.

- 발급받은 `igibson.key`를 데이터 루트에 위치
- `ig_dataset`을 다운로드/압축해제하여 위 경로에 배치

버전에 따라 자동 다운로드 명령이 가능한 경우:

```bash
python -m igibson.utils.assets_utils --download_ig_dataset
```

만약 위 명령이 동작하지 않으면, 공식 문서의 안내대로 수동 다운로드/압축해제를 사용하세요.

### 2-4. 빠른 경로 점검

```bash
ls -al /mount/nas2/users/dukim/vla_ws/igibson/data
ls -al /mount/nas2/users/dukim/vla_ws/igibson/data/assets | head
ls -al /mount/nas2/users/dukim/vla_ws/igibson/data/ig_dataset | head
```

---

## 3. Entry Point 실행 (`main_llm_zeroshot.py`)

### 3-1. 일반 실행 (debug vis 없음)

```bash
python main_llm_zeroshot.py \
  --use_igibson 1 \
  --num_processes 1 \
  --no_cuda \
  --use_gtsem 1
```

### 3-2. Debug Visualization 포함 실행

```bash
python main_llm_zeroshot.py \
  --use_igibson 1 \
  --num_processes 1 \
  --no_cuda \
  --use_gtsem 1 \
  --debug_viz 1 \
  --debug_viz_dir ./tmp/debug_viz_ig \
  --debug_viz_every 1
```

출력 폴더:

```text
./tmp/debug_viz_ig/
  obs/
  maps/local/
  maps/planner/
  maps/frontier/
  meta/
```

---

## 4. 자주 발생하는 문제

- 데이터셋 경로를 못 찾는 경우:
  - `envs/__init__.py`의 기본 경로와 실제 파일 위치가 동일한지 확인
  - 특히 `igibson.key`, `assets`, `ig_dataset` 누락 여부 점검

- 렌더링/X11 문제:
  - `docker-compose.yml`의 `DISPLAY`, `XAUTHORITY` 설정 확인
  - 헤드리스로 사용할 경우 iGibson 렌더링 모드 설정을 별도로 점검

---

## 5. 참고 링크

- iGibson 설치 문서: https://stanfordvl.github.io/iGibson/installation.html
- iGibson 데이터셋 문서: https://stanfordvl.github.io/iGibson/dataset.html
- iGibson GitHub: https://github.com/StanfordVL/iGibson
