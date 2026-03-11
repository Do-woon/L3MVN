# L3MVN → iGibson Porting Notes — v2.2 (책임 분리 규약 반영)

> 업데이트 내용(v2):
> - TODO#4: iGibson GT semantic의 (semantic_id → category name) 경로를 확정하고,
>   L3MVN `hm3d_category`(15-class)로의 매핑 초안을 제안한다.
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

## 12. 매핑 설계 원칙 (실무적인 관점)

iGibson의 카테고리(= `categories.txt` 기반)는 매우 세분화될 수 있으므로, 포팅 v1 목표(“루프 검증”)를 위해 다음 원칙을 사용한다.

- **원칙 A: L3MVN 15-class로 “축약”**한다. (세부 물체는 상위 클래스에 흡수)
- **원칙 B: substring / synonym 기반 매핑**을 먼저 쓰고(빠른 검증),
  이후 필요하면 categories.txt를 기반으로 규칙을 보강한다.
- **원칙 C: 애매한 물체는 `appliances`로 흡수하거나, 미매핑(=ignore) 처리한다.
  (v1에서는 recall보다 안정성이 중요)

---

## 13. 매핑 규칙(초안)

아래는 iGibson class name(문자열)을 입력으로 받아 L3MVN class index(1..15)를 내는 규칙 초안이다.
(16번째 채널에 대한 매핑은 정의하지 않으며, 해당 채널은 비워둔다.)

### 13.1 Direct / synonym 매핑
| iGibson class name contains | L3MVN class |
|---|---|
| "chair", "armchair", "stool", "bench", "highchair" | chair |
| "sofa", "couch", "loveseat" | sofa |
| "plant", "potted", "flower", "vase_plant" | plant |
| "bed" | bed |
| "toilet" | toilet |
| "tv", "television", "monitor", "screen" | tv_monitor |
| "bathtub", "tub" | bathtub |
| "shower" | shower |
| "fireplace" | fireplace |
| "towel" | towel |
| "sink", "basin" | sink |
| "dresser", "chest_of_drawers", "drawer", "cabinet_dresser" | chest_of_drawers |
| "table", "desk", "counter", "coffee_table", "dining_table" | table |
| "stairs", "stair" | stairs |

### 13.2 Appliances(흡수) 매핑
다음은 iGibson에서 흔히 등장 가능한 “가전/설비”를 `appliances`로 흡수한다.
| iGibson class name contains | L3MVN class |
|---|---|
| "refrigerator", "fridge", "microwave", "oven", "dishwasher", "washer", "dryer", "stove", "cooktop" | appliances |

### 13.3 Ignore / background
- `agent`, `robot` 등 로봇 자체 클래스는 ignore (L3MVN semantic map의 object 채널로 넣지 않음)
- `background` 또는 0 class는 ignore
- DIRT/STAIN/WATER 등 특수 마커는 ignore (v1)

> 위 ignore 규칙은 iGibson의 `SemanticClass`에 의해 존재할 수 있는 특수 id들을 고려한 것이다. :contentReference[oaicite:14]{index=14}

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

## 15. 남는 불확실성 (하지만 “코드 없이”도 계획으로는 충분한 수준)

- iGibson `categories.txt`의 실제 항목(정확한 문자열 목록)은 설치된 iG dataset 버전에 의존한다.
- 따라서 매핑 규칙은 “초안”이며, 포팅 v1에서 다음을 통해 안정화한다:
  - 실제 런타임에서 관측되는 상위 빈도 class name top-K를 로그로 뽑아,
  - 규칙(13.x)을 보강/예외처리한다.

(즉, 계획 단계에서는 지금 수준으로 충분히 문서화가 가능하고,
구현 단계에서 데이터 기반으로 refine하는 것이 합리적이다.)

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
