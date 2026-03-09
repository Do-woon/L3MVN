# L3MVN iGibson Docker Guide

`docker/` 디렉토리는 iGibson 포팅 개발용 컨테이너 설정을 담고 있습니다.

## 파일 구성

- `Dockerfile`: CUDA/PyTorch/iGibson 기반 개발 이미지 정의
- `.dockerignore`: 이미지 빌드 컨텍스트에서 제외할 파일 목록

## 사전 조건

- Docker + NVIDIA Container Toolkit 사용 가능 환경
- 호스트 경로 `/mount/nas2`, `/mount/nas3` 접근 가능
- 프로젝트 루트 경로: `/mount/nas2/users/dukim/vla_ws/L3MVN`

## 실행 절차

프로젝트 루트에서 실행:

```bash
docker-compose build
docker-compose up -d
docker-compose exec l3mvn bash
```

작업 종료:

```bash
docker-compose down
```

## 기본 스펙

- Base image: `nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04`
- Python: 3.8
- PyTorch: 1.7.0 (CUDA 11.0 wheel)
- iGibson: GitHub `StanfordVL/iGibson` editable install

## 참고

- 컨테이너 내부 작업 디렉토리: `/mount/nas2/users/dukim/vla_ws/L3MVN`
- iGibson 사용자 설정은 Docker volume `igibson_config`에 저장됩니다.
