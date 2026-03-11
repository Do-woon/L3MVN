# L3MVN → iGibson Porting Notes — v2.3 (curated taxonomy 반영)

> 업데이트 내용(v2):
> - TODO#4: iGibson GT semantic의 (semantic_id → category name) 경로를 확정하고,
>   L3MVN `hm3d_category`(15-class)로의 curated 매핑 규칙을 정리한다.
>
> 업데이트 내용(v2.1 — 코드 대조 검증 반영):
> - Section 14: semantic one-hot 채널 수 **15 → 16**으로 수정
> - Section 14: obs 전체 채널 수 **19 → 20**으로 수정
> - num_sem_categories=16 확인 반영
>
> 업데이트 내용(v2.2 — 최종 규약 통일):
> - iGibson adapter는 semantic id single-channel을 전달한다는 점 명시
> - 16채널 one-hot 생성 책임은 `Sem_Exp_Env_Agent._preprocess_obs()`로 명시
>
> 업데이트 내용(v2.3 — curated taxonomy 반영):
> - SemanticTaxonomy를 substring 중심 임시 규칙에서 curated alias mapping 중심으로 전환
> - 애매한 클래스(`cabinet`, `bookshelf`, `tv_stand` 등)는 의도적으로 0(unmapped) 유지
> - 분포/top-K 기반 자동 규칙 확장 문구 제거
>
> 근거 소스:
> - iGibson seg/ins_seg 언노말라이즈 및 MAX_CLASS_COUNT/MAX_INSTANCE_COUNT: iGibson Renderer 문서.
> - MAX_CLASS_COUNT=512, MAX_INSTANCE_COUNT=1024 정의: `igibson/utils/constants.py`.
> - iGibson class name ↔ class id 매핑 생성: `igibson/utils/semantics_utils.py` (`categories.txt`를 읽어 연속 class id 부여).
> - L3MVN의 15개 semantic category 정의: `L3MVN/constants.py`의 `hm3d_category`.
> - L3MVN의 semantic one-hot 생성: `agents/sem_exp.py:390-392` (16채널 확인)
> - L3MVN의 num_sem_categories 기본값: `arguments.py:137` (16 확인)

---

## 10. iGibson GT semantic: 값의 의미와 “id → name” 경로 확정

### 10.1 iGibson `seg` 이미지의 저장 형식 (렌더러)
- iGibson 렌더러에서 `modes=('seg','ins_seg')`로 렌더하면 4채널 이미지가 오고,
  **첫 번째 채널(seg[:,:,0])이 semantic class id**(정규화된 값)이다. :contentReference[oaicite:4]{index=4}
- 값은 [0,1]로 정규화되어 있으며,
  - semantic seg: `MAX_CLASS_COUNT = 512`
  - instance seg: `MAX_INSTANCE_COUNT = 1024`
  로 **정수 id로 복원**한다. :contentReference[oaicite:5]{index=5}

복원 공식(문서에 제시):
- `seg_id = (seg[:, :, 0:1] * MAX_CLASS_COUNT).astype(np.int32)` :contentReference[oaicite:6]{index=6}
- `ins_id = (ins_seg[:, :, 0:1] * MAX_INSTANCE_COUNT).astype(np.int32)` :contentReference[oaicite:7]{index=7}

> 참고: iGibsonEnv에서 VisionSensor를 쓰면 `get_seg()` / `get_ins_seg()` 호출 시 이 언노말라이즈를 “직접 수행한다”고 문서에 명시되어 있다. :contentReference[oaicite:8]{index=8}

### 10.2 iGibson “class id → class name” 매핑의 소스
iGibson은 `igibson/utils/semantics_utils.py`에서
- `igibson.ig_dataset_path/metadata/categories.txt`를 읽고 :contentReference[oaicite:9]{index=9}
- scene object class id를 **연속 정수**로 배정한다. (기본 `starting_class_id = SemanticClass.SCENE_OBJS + 1`) :contentReference[oaicite:10]{index=10}
- 결과를 `CLASS_NAME_TO_CLASS_ID`로 전역 보관한다. :contentReference[oaicite:11]{index=11}

즉, iGibson에서 GT semantic을 name으로 바꾸는 “정식 경로”는:

1) `seg_id` (int32) 추출 (10.1)
2) `CLASS_NAME_TO_CLASS_ID = get_class_name_to_class_id()` (10.2)
3) 이를 뒤집어 `CLASS_ID_TO_CLASS_NAME = {v:k for k,v in CLASS_NAME_TO_CLASS_ID.items()}`
4) 각 픽셀의 `seg_id`를 `CLASS_ID_TO_CLASS_NAME[seg_id]`로 lookup (미정의 id는 background/agent 등 예외처리)

> 주의: iGibson은 semantic id에 “scene objects” 외에도 background/robot 및 특수 마커(예: DIRT/STAIN/WATER 등)를 포함한다. 이 예약/특수 id 개념은 `SemanticClass` enum 및 상수 정의에서 확인 가능. :contentReference[oaicite:12]{index=12}

---

## 11. iGibson category name → L3MVN `hm3d_category`(15) 매핑 초안

### 11.1 L3MVN이 기대하는 15개 semantic category
L3MVN의 로컬 semantic map 채널(4+)이 가정하는 클래스 문자열은 다음 15개다. :contentReference[oaicite:13]{index=13}

1. chair
2. sofa
3. plant
4. bed
5. toilet
6. tv_monitor
7. bathtub
8. shower
9. fireplace
10. appliances
11. towel
12. sink
13. chest_of_drawers
14. table
15. stairs

---

## 12. 매핑 설계 원칙 (curated mapping 중심)

iGibson의 카테고리(= `categories.txt` 기반)는 매우 세분화되어 있으므로,
taxonomy는 다음 원칙으로 고정한다.

- **원칙 A: curated alias table이 1순위다.**  
  `category -> aliases`를 명시적으로 관리하고 deterministic하게 매핑한다.
- **원칙 B: class name 정규화 후 exact alias 매칭이 기본이다.**  
  (`lowercase`, 구분자 정규화, WordNet suffix 제거)
- **원칙 C: fallback은 최소화한다.**  
  multi-token alias에 대한 제한적 토큰 연속 매칭만 허용한다.
- **원칙 D: 애매한 클래스는 0(background/ignore)로 유지한다.**  
  coverage보다 일관성과 오매핑 방지를 우선한다.

---

## 13. Curated alias 매핑 규칙(최종)

아래는 iGibson class name(문자열)을 입력으로 받아 L3MVN class index(1..15)를 내는
curated alias 규칙이다.
(16번째 채널은 `_preprocess_obs()`의 one-hot 생성 규약 채널이며 taxonomy가 직접 쓰지 않는다.)

### 13.1 Category별 curated alias
| L3MVN class | curated aliases (대표) |
|---|---|
| chair | chair, armchair, office_chair, folding_chair, straight_chair, swivel_chair, rocking_chair, stool, bench, highchair |
| sofa | sofa, chaise_longue, couch, loveseat, sectional, futon |
| plant | plant, potted_plant, pot_plant, flower, vase_plant |
| bed | bed, bunk_bed, crib |
| toilet | toilet |
| tv_monitor | tv, television, monitor, standing_tv, wall_mounted_tv, screen |
| bathtub | bathtub, tub |
| shower | shower |
| fireplace | fireplace |
| appliances | refrigerator, fridge, microwave, oven, dishwasher, washer, dryer, stove, griddle, grill, heater, iron, kettle, range_hood, toaster, vacuum, burner, blender, coffee_maker, cooktop |
| towel | towel, bath_towel, hand_towel, dishtowel, rag |
| sink | sink, basin, bathroom_sink, kitchen_sink |
| chest_of_drawers | dresser, drawer, chest_of_drawers, bureau, cabinet_dresser |
| table | table, breakfast_table, dining_table, coffee_table, side_table, console_table, desk, gaming_table, pedestal_table, pool_table, counter, countertop, kitchen_counter |
| stairs | stair, stairs |

### 13.2 의도적 unmapped(0) 클래스
- `agent`, `robot` 등 로봇 자체 클래스는 ignore (L3MVN semantic map의 object 채널로 넣지 않음)
- `background` 또는 0 class는 ignore
- DIRT/STAIN/WATER 등 특수 마커는 ignore (v1)
- 보수적 정책으로 다음 클래스도 기본적으로 0 유지:
  - `cabinet` (일반형), `bookshelf`, `bookcase`, `shelf`, `rack`, `tv_stand`, `ottoman`

> 위 ignore 규칙은 iGibson의 `SemanticClass`에 의해 존재할 수 있는 특수 id들을 고려한 것이다. :contentReference[oaicite:14]{index=14}

### 13.3 수동 검토 기반 최종 추가 alias (categories.txt 1:1 검토 반영)
`categories.txt` 1..394 전 항목을 수동 검토한 결과를 taxonomy에 반영했다.

- 신규/보강 alias 예시:
  - chair: `straight_chair`, `swivel_chair`, `rocking_chair`
  - sofa: `chaise_longue`
  - plant: `pot_plant`
  - bed: `crib`
  - tv_monitor: `standing_tv`, `wall_mounted_tv`
  - appliances: `griddle`, `grill`, `heater`, `iron`, `kettle`, `range_hood`, `toaster`, `vacuum`, `burner`, `blender`, `coffee_maker`
  - towel: `dishtowel`, `rag`
  - table: `gaming_table`, `pedestal_table`, `pool_table`

참고용 현재 taxonomy 기준 분포, (runtime class 395개):
- `sem_id=0`: 345
- `sem_id=1`: 8, `2`: 2, `3`: 1, `4`: 2, `5`: 1, `6`: 3, `7`: 1, `8`: 1
- `sem_id=10`: 18, `11`: 4, `12`: 1, `14`: 8

---

## 14. Stage 1 semantic 전달 vs Stage 2 one-hot 생성 (책임 분리)

L3MVN mapper는 `obs[:, 4:, :, :]`를 semantic one-hot(or prob) 채널로 기대한다.  
다만 최종 규약에서 **생성 책임 위치**는 다음처럼 고정한다:

- iGibson adapter는 semantic id single-channel을 전달한다.
- 16채널 one-hot은 L3MVN preprocessing 단계에서 생성된다.

즉 taxonomy(Section 10~13)는 semantic id를 해석하는 규칙을 제공하고,
실제 one-hot 텐서 생성은 `Sem_Exp_Env_Agent._preprocess_obs()`가 수행한다.

**핵심 수치 (코드 검증 결과):**
- `args.num_sem_categories` = **16** (arguments.py:137)
- `sem_seg_pred` shape = **(H, W, 16)** (agents/sem_exp.py:390)
- map 총 채널 = `num_sem_categories + 4` = **20** (main_llm_zeroshot.py:139)
- hm3d_category는 15개이지만, one-hot은 **16채널**로 생성됨 (15 named + 1 extra)

실제 one-hot 생성 위치 (agents/sem_exp.py:390-392):
```python
sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))  # 16 channels
for i in range(16):
    sem_seg_pred[:,:,i][semantic == i+1] = 1
```

iGibson 포팅에서의 단계별 책임:

1) Adapter/Env 단계 (Stage 1)
   - `seg_id` (H,W) int32 획득 (10.1)
   - `class_id → class_name` 변환 (10.2)
   - `class_name → L3MVN_idx(1..15)` 해석 규칙 적용 (13.x)
   - 최종 전달 포맷: semantic id single-channel (`(1,H,W)` 또는 Stage 1 `(5,H,W)`의 마지막 채널)
2) Agent preprocessing 단계 (Stage 2)
   - `_preprocess_obs()`가 semantic id를 16채널 one-hot으로 변환
   - `_preprocess_obs()`가 depth 전처리(cm 변환 포함)도 수행
   - 최종 obs/state: **[RGB(3), Depth(1), Sem(16)] = 20채널**

> 주의: main loop에서 frontier semantic summary를 추출할 때는
> `range(args.num_sem_categories-1)` = `range(15)`, 즉 채널 4..18 (15개)만 순회하고
> 채널 19 (16번째)는 사용하지 않음 (main_llm_zeroshot.py:739).
> 그러나 Semantic_Mapping 모듈은 `obs[:, 4:, :, :]`로 **16채널 전부** 사용하므로 (model.py:88-90),
> 반드시 16채널을 맞춰야 함.

---

## 15. 남는 제약 / 운영 원칙

- iGibson `categories.txt`의 실제 항목은 설치된 dataset 버전에 의존한다.
- 본 규약에서는 분포/top-K 기반 자동 확장은 사용하지 않는다.
- 신규 클래스 대응이 필요하면 curated alias table을 수동 갱신한다.

---

## 16. L3MVN 목표 카테고리(`category_to_id`) vs iGibson 매핑 (v2.1 추가)

L3MVN에서 LLM scoring 대상 목표 카테고리는 `hm3d_category`(15개)가 **아니라**
`category_to_id`(6개)이다 (constants.py:19-26):

```python
category_to_id = [“chair”, “bed”, “plant”, “toilet”, “tv_monitor”, “sofa”]
```

이 6개는 ObjectNav 태스크에서 탐색 대상이 되는 목표 객체 카테고리이며:
- `construct_dist(objs_list)`가 이 6개에 대해 LLM 점수를 산출 (main_llm_zeroshot.py:478)
- `infos['goal_name']`도 이 리스트의 원소여야 함

> iGibson 이식 시:
> - iGibson에서 지원하는 ObjectNav 목표 카테고리를 이 6개에 매핑해야 함
> - 6개 모두 iGibson에 존재하는 객체 클래스인지 확인 필요
> - 존재하지 않는 카테고리가 있으면 `category_to_id` 리스트 자체를 수정하거나,
>   iGibson 씬에서 해당 객체가 포함된 에피소드만 필터링해야 함
