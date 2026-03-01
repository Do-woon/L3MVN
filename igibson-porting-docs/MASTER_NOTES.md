# L3MVN → iGibson Porting Notes (Interface Inventory) — v0.1

> 목표: L3MVN을 iGibson으로 이식하기 위해, “상호작용 인터페이스”를 먼저 식별하고
> (L3MVN 기대값) vs (iGibson 제공값) 차이를 기록한다.
>
> 전제:
> 1) zero-shot LLM scoring 사용
> 2) GT semantic 사용
> 3) 이산 이동(Forward/Turn/Stop/Look Up/Down)으로 래핑
>
> v0.1 변경사항 (코드 대조 검증 반영):
> - 액션 공간: 4개 → **6개** (Look Up/Down 추가)
> - depth 전처리 후 단위: meters → **centimeters**
> - obs 채널: (H,W,5) 원시 / (C,H,W) 전처리 후 **20채널** 구분 명시
> - infos 필수 키: sensor_pose 외 eve_angle, goal_cat_id, goal_name, clear_flag 추가
> - Semantic_Mapping 반환값 4개 명시 (translated, map_pred, map_pred_stair, current_poses)
> - plan_act_and_preprocess 반환값: reward → **fail_case** (dict) 수정
> - category_to_id(6개) vs hm3d_category(15개) 구분 명시
> - remove_small_points의 Goal_score 반환값 문서화

---

## 1. 시스템 경계도 (최상위 호출 흐름)

### 1.1 Entry script
- `main_llm_zeroshot.py` : zero-shot frontier scoring 루프 (이번 이식의 주 target)
- `main_llm_vis.py` : feed-forward/LM embedding + (옵션) pretrained 로드 루프

### 1.2 Environment creation entry point
- `envs.make_vec_envs(args)` 가 유일한 “환경 생성” 엔트리로 보며, 이 함수가 반환하는 객체는:
  - `reset() -> (obs_tensor, infos)`
  - `plan_act_and_preprocess(planner_inputs) -> (obs_tensor, fail_case, done, infos)`
    - **주의**: 두 번째 반환값은 reward가 아니라 `fail_case` (dict). VecPyTorch에서 torch 변환 없이 pass-through됨. (envs/__init__.py:53-57)
  - `step(actions)` 등도 존재하나, main 루프는 주로 `plan_act_and_preprocess`를 사용

> iGibson 이식 시 **가장 먼저** 교체/추상화할 접점은 `make_vec_envs`와 그 내부에서 감싸는 `VecPyTorch` 래퍼임.

---

## 2. L3MVN ↔ Env 인터페이스 명세 (현 레포 관찰 기반)

### 2.1 Reset contract
- Call: `obs, infos = envs.reset()`
- obs: torch.FloatTensor on device (VecPyTorch가 numpy -> torch 변환)
- infos: python list/dict (env thread별 info dict)

**확정 사항 (코드 검증 완료)**
- obs shape: `(N, C, H, W)` where C = 20 (전처리 후: RGB 3 + Depth 1 + Semantic one-hot 16)
- infos[env_idx] 필수 키:
  - `sensor_pose`: [dx, dy, do] 상대 delta pose (meters, radians). sem_map_module 입력. (main_llm_zeroshot.py:594-597)
  - `eve_angle`: elevation angle (정수, 0/-30/-60). sem_map_module 5번째 인자. (main_llm_zeroshot.py:599-601)
  - `goal_cat_id`: 목표 semantic category index (int). local_map 채널 참조용 (cn = goal_cat_id + 4). (main_llm_zeroshot.py:728)
  - `goal_name`: 목표 category 문자열. LLM scoring 시 category_to_id.index(cname)으로 사용. (main_llm_zeroshot.py:729)
  - `clear_flag`: 맵 초기화 트리거 (0 또는 1). (main_llm_zeroshot.py:672)
- episode 종료 시 추가 필수 키:
  - `spl`, `success`, `distance_to_goal`: 평가 메트릭. (main_llm_zeroshot.py:576-578)
- 조건부 키:
  - `g_reward`: Gibson task_config일 때만 사용. (main_llm_zeroshot.py:813)

### 2.2 plan_act_and_preprocess contract
- Call: `obs, fail_case, done, infos = envs.plan_act_and_preprocess(planner_inputs)`
- planner_inputs: dict (env 개수만큼의 list). 필수 키는 L3MVN_ANALYSIS.md Section 2 참고.
- obs: 다음 step 관측치 (torch.FloatTensor on device)
- **fail_case**: dict (`{'collision': int, 'exploration': int, 'detection': int, 'success': int}`). reward가 아님. (agents/sem_exp.py:76-80, 200)
- done: bool ndarray (env thread별 종료)
- infos: list[dict] (metric, pose 등)

**확정 사항**
- planner_inputs의 필수 키: `map_pred`, `exp_pred`, `pose_pred`, `goal`, `map_target`, `new_goal`, `found_goal`, `wait` (상세 spec은 L3MVN_ANALYSIS.md Section 2)
- done 처리: episode 종료 시 `init_map_and_pose_for_env(e)`를 **main loop가** 호출 (main_llm_zeroshot.py:589)

---

## 3. Observation Tensor Layout (가장 중요한 “형태 계약”)

L3MVN에는 **두 단계의 obs 형태**가 존재한다:

### 3.0 원시 env 출력 vs 전처리 후 state (중요 구분)

**Stage 1: 원시 env 출력** (ObjectGoal_Env/ObjectGoal_Env21의 reset/step 반환)
- `state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)`
- shape: `(C, H, W)` where C = 3 (RGB) + 1 (Depth) + 1 (Semantic ID) = **5**
- 이 5채널 텐서가 VecPyTorch를 거쳐 main loop에 도달

**Stage 2: 전처리 후 state** (Sem_Exp_Env_Agent._preprocess_obs 이후)
- `_preprocess_obs`가 Stage 1 텐서를 받아 (H,W,C)로 transpose 후:
  - `rgb = obs[:, :, :3]`
  - `depth = obs[:, :, 3:4]`
  - `semantic = obs[:, :, 4:5].squeeze()` (GT semantic id map)
- GT semantic 경로 (`use_gtsem=True`):
  - `sem_seg_pred = np.zeros((H, W, 15 + 1))` → **16채널 one-hot**
  - `for i in range(16): sem_seg_pred[:,:,i][semantic == i+1] = 1`
- depth 전처리 후 재조합:
  - `state = concat(rgb(3), depth(1), sem_seg_pred(16))` → `(C,H,W)` 변환
  - 최종 shape: `(20, H, W)` = **20채널**

> **Semantic_Mapping 모듈은 Stage 2 (20채널)을 입력으로 받는다.**
> ch0..2=RGB, ch3=Depth, ch4..19=Semantic one-hot(16ch).

### 3.1 GT semantic 사용 시 기대 포맷
- input `semantic`은 정수 semantic id map (H,W)
- code는 `for i in range(16): sem_seg_pred[:,:,i][semantic == i+1] = 1` (agents/sem_exp.py:391-392)
  - GT semantic id 1..16이 one-hot 채널 0..15에 매핑됨
- sem_seg_pred shape: `(H,W, 16)` = **15 named categories + 1 extra channel**
  - 채널 0..14: hm3d_category[0..14] (“chair”~”stairs”)
  - 채널 15: 추가 채널 (코드상 id=16에 대응, 실제 사용 빈도 낮음)
- `args.num_sem_categories` 기본값 = **16** (arguments.py:137)
- map 총 채널 = `num_sem_categories + 4` = **20** (main_llm_zeroshot.py:139)
  - ch0: obstacle, ch1: explored, ch2: current agent, ch3: past agent, ch4..19: semantic

> iGibson GT semantic id → (L3MVN 내부에서 기대하는) 1..16 id 로 매핑 필요.
> 15개 named category에 대해 1..15를 부여하고, 16번 채널은 여분으로 둔다.
> 이 매핑은 “taxonomy adapter”로 분리.

### 3.2 Depth 전처리 기대
- `_preprocess_depth`는 depth에서 0, 0.99 등의 값을 치환/클리핑함. (agents/sem_exp.py:425-440)
- 전처리 공식: `depth = min_d * 100.0 + depth * (max_d - min_d) * 100.0`
  - 기본값: min_d=0.5m, max_d=5.0m
  - 입력: Habitat normalized depth [0, 1]
  - **출력: 센티미터(cm) 단위** (50~500cm 범위)
- Semantic_Mapping 모듈의 `du.get_point_cloud_from_z_t`가 이 cm 단위 depth를 받아 point cloud 생성
- iGibson adapter에서는:
  - iGibson depth (meters) → Habitat-style normalized [0,1]로 변환 후 `_preprocess_depth` 재사용
  - 또는 iGibson depth (meters) → 직접 cm로 변환하여 `_preprocess_depth` 우회

### 3.3 Downscale 규약
- `ds = env_frame_width // frame_width` (정수 downscale)
- ds != 1 이면 rgb resize + depth/semantic stride sampling 수행

> iGibson에서 센서 해상도와 L3MVN 내부 해상도(frame_width/height) 분리 운영 가능.
> 단, ds가 정수로 떨어지도록 맞추는 것이 구현/디버깅이 쉬움.

---

## 4. Map / Planner 모듈 접점

### 4.1 Semantic mapping module
- `model.Semantic_Mapping.forward(obs, pose_obs, maps_last, poses_last, eve_angle)` (model.py:63)
- 내부에서 `depth = obs[:, 3, :, :]` (model.py:65)
- 내부에서 semantic feature는 `obs[:, 4:, :, :]` → **16채널** (model.py:88-90)
- 5번째 인자 `eve_angle`: infos에서 추출한 elevation angle 배열 (model.py:71)

**반환값** (model.py:229):
```
translated, map_pred, map_pred_stair, current_poses
```
main loop에서의 변수명:
```python
increase_local_map, local_map, local_map_stair, local_pose = \
    sem_map_module(obs, poses, local_map, local_pose, eve_angle)
```
- `local_map_stair`: 계단 처리 로직에 사용 (main_llm_zeroshot.py:629-630)

**따라서 필수 계약**
- obs tensor channel semantics (20채널):
  - ch0..2: RGB (또는 placeholder)
  - ch3: Depth (**cm 단위**, 전처리 후)
  - ch4..19: Semantic (**16채널**, one-hot)

### 4.2 MapBuilder (legacy) 사용 여부
- `envs/utils/map_builder.py`는 depth + pose로 occupancy/explored를 업데이트하는 모듈
- L3MVN main loop는 Semantic_Mapping(토치 모듈)을 주로 쓰는 것으로 보이나,
  코드 일부에서 MapBuilder 또는 유사 기능을 병용할 수 있음.

> iGibson 이식 v1에서는 MapBuilder를 “참고 구현”으로만 두고,
> 실제 main loop가 쓰는 Semantic_Mapping 입력 계약을 우선 만족시키는 것이 합리적.

### 4.3 FMMPlanner contract (로컬 이동)
- `FMMPlanner(traversible, scale=1, step_size=5)`
- `set_multi_goal(goal_map)` : goal_map==1인 cell을 목표로 distance transform
- `get_short_term_goal(state)` : (x,y)에서 단기 목표, replan, stop 반환

**이산 이동 래핑과의 연결**
- FMMPlanner가 내놓는 단기 목표는 “grid 상의 좌표”
- env는 이를 실제 로봇/시뮬 이동(연속)으로 실행해야 함
- 따라서 iGibson 쪽에는:
  - (옵션 A) L3MVN이 기대하는 기존 방식대로 “FMM → discrete action”까지 env가 제공
  - (옵션 B) L3MVN이 단기 goal (x,y)만 주고, iGibson adapter가 이를 waypoint-follow로 실행
- 이번 계획에서는 (3) 이산 이동 래핑을 하므로, 옵션 A에 가깝게 설계

---

## 5. Zero-shot LLM Scoring 접점

### 5.1 Frontier description 생성 위치
- frontier 후보를 만들고
- 각 frontier 주변의 semantic object set을 추출해
- LLM prompt input으로 구성
- LLM output(점수/순위)을 frontier selection에 반영

**문서화 필요 항목**
- frontier 후보 데이터 구조:
  - 위치 표현: (grid row,col)인지, metric (x,y)인지
  - 주변 semantic summary: “객체 리스트”, “룸 타입”, “거리” 등 어떤 필드를 넣는지
- LLM scoring interface:
  - 입력: prompt string (또는 list)
  - 출력: frontier score list (float)

> iGibson 이식 v1에서는 prompt 구성은 최대한 그대로 유지하고,
> “frontier 주변 semantic summary 추출”만 iGibson taxonomy에 맞춰 재현한다.

---

## 6. Habitat 결합부(제거 대상) 목록

iGibson 이식 범위에서는 아래 Habitat 전용 모듈은 “대체/우회” 대상:
- `envs/habitat/*` (dataset loading, VectorEnv, ObjectGoal_Env 등)
- `agents/sem_exp.py` 의 base class가 Habitat env에 결합되어 있음

단, `agents/sem_exp.py` 안의 다음 로직은 재사용 가능성이 높음:
- `_preprocess_obs` (GT semantic 경로)
- depth 전처리, semantic one-hot 생성, downscale 규약
- 시각화/디버깅 루틴(옵션)

> 전략: “Habitat env 클래스 상속 구조”는 버리고,
> SemExp가 기대하는 관측→state 생성 함수만 iGibson adapter로 옮겨오는 방식이 안전.

---

## 7. iGibson Adapter 설계 초안 (문서화 대상)

### 7.1 igibson_adapter.EnvWrapper (VecPyTorch 호환)
필수 메서드:
- `reset() -> (obs_np, infos_list)`
- `plan_act_and_preprocess(planner_inputs) -> (obs_np, fail_case, done_np, infos_list)`
  - 두 번째 반환값은 reward가 아닌 `fail_case` dict

infos_list 각 원소가 반드시 포함해야 하는 키:
- `sensor_pose`: [dx, dy, do] (meters, radians) — 상대 delta pose
- `eve_angle`: elevation angle (0, -30, -60 등) — sem_map_module 입력
- `goal_cat_id`: 목표 semantic category index (int)
- `goal_name`: 목표 category 문자열 (hm3d_category 또는 category_to_id의 원소)
- `clear_flag`: 맵 초기화 트리거 (0 or 1)
- (episode 종료 시) `spl`, `success`, `distance_to_goal`

### 7.2 igibson_adapter.ObsAdapter
- iGibson raw sensors:
  - rgb: (H,W,3) uint8
  - depth: (H,W,1) float32 meters
  - semantic_id: (H,W) uint32/int (GT)
- 출력:
  - L3MVN가 기대하는 obs tensor로 pack:
    - (H,W, 3 + 1 + 1) 혹은 (H,W, 3+1+1) 형태의 “raw obs”
    - 또는 바로 (C,H,W) state로 변환해 반환 (선호: 기존 SemExp `_preprocess_obs` 계약에 맞추기)

### 7.3 igibson_adapter.SemanticTaxonomy
- iGibson semantic id -> canonical name 매핑 확보
- canonical name -> L3MVN category index(1..K) 매핑 정의
- 목표물(goal)도 동일 taxonomy 상의 문자열로 표현

### 7.4 igibson_adapter.DiscreteActionExecutor
- actions: **6개 이산 액션** (agents/sem_exp.py:302-336)

| Action ID | 이름 | 동작 |
|---|---|---|
| 0 | STOP | 정지 (episode 종료 트리거) |
| 1 | FORWARD | 전진 (forward_step_m만큼) |
| 2 | TURN_LEFT | 좌회전 (turn_angle_deg만큼) |
| 3 | TURN_RIGHT | 우회전 (turn_angle_deg만큼) |
| **4** | **LOOK_UP** | **카메라 elevation +30도** (eve_angle += 30) |
| **5** | **LOOK_DOWN** | **카메라 elevation -30도** (eve_angle -= 30) |

- parameter:
  - forward_step_m (예: 0.25m)
  - turn_angle_deg (예: 30deg, arguments.py:81 기본값)
  - eve_angle_step_deg: 30deg (elevation 변경 단위, agents/sem_exp.py:326-329)
  - eve_angle 범위: 0 ~ -60 (최소 -60도까지 내려봄, agents/sem_exp.py:324)
- 출력:
  - iGibson 제어 입력(연속/속도 제어 등)을 사용해 해당 macro를 실행하고,
  - 실제 실행 후 pose delta를 infos['sensor_pose']에 기록
  - elevation 변경 시 infos['eve_angle']을 업데이트

> **주의**: Look Up/Down (4,5)은 로봇 이동이 아니라 카메라 tilt 변경이므로,
> sensor_pose delta는 [0, 0, 0]이 되어야 하고, eve_angle만 변경해야 한다.
> 현재 코드에서 eve_angle은 env 내부 상태로 관리됨 (agents/sem_exp.py:107-108).

---

## 8. Gap Table 템플릿 (다음 단계에서 채울 것)

| Interface | L3MVN expects | iGibson provides | Gap | Adapter 책임 |
|---|---|---|---|---|
| RGB | uint8 (H,W,3) | ? | - | ObsAdapter |
| Depth | normalized [0,1] → 전처리 후 **cm** | ? | 단위 변환 필요 | ObsAdapter |
| Semantic GT | int id map (1..16), 16ch one-hot | uint32 id map + mapping dict | id space mismatch | SemanticTaxonomy |
| Pose (sensor_pose) | [dx,dy,do] delta (meters, radians) | base pose + camera extrinsic | frame/delta 변환 | ObsAdapter (pose) |
| Pose (eve_angle) | elevation angle (0/-30/-60) | camera tilt state | elevation 상태 관리 | DiscreteActionExecutor |
| Action | **discrete(6)**: Stop/Fwd/Left/Right/**LookUp/LookDown** | continuous base ctrl | macro execution 필요 | DiscreteActionExecutor |
| infos keys | sensor_pose, eve_angle, goal_cat_id, goal_name, clear_flag, metrics | N/A | 전부 생성 필요 | EnvWrapper |
| planner_inputs | dict 8 keys (Section 2 참조) | N/A | define contract | EnvWrapper |

---

## 9. Immediate Questions to Answer by Inspection (no coding yet)

1) `planner_inputs`에는 어떤 키들이 필수인가? → **해결됨** (L3MVN_ANALYSIS.md Section 2)
2) `infos`에 반드시 있어야 하는 key는? → **해결됨** (위 Section 2.1, 총 6개 필수 + 3개 조건부)
3) frontier set의 내부 표현은? → **해결됨** (L3MVN_ANALYSIS.md Section 4)
4) success/stop 조건은 main loop가 판단하는가, env가 done으로 주는가?
   → **env가 done을 반환**하고, main loop는 done 시 `init_map_and_pose_for_env(e)` 호출 및 메트릭 기록

---

## 10. LLM Scoring에서의 category_to_id vs hm3d_category 구분 (추가)

L3MVN에는 두 개의 서로 다른 category 리스트가 존재한다:

### 10.1 `hm3d_category` (15개) — semantic map 채널 매핑용
```python
hm3d_category = ["chair", "sofa", "plant", "bed", "toilet", "tv_monitor",
                  "bathtub", "shower", "fireplace", "appliances", "towel",
                  "sink", "chest_of_drawers", "table", "stairs"]
```
- local_map 채널 4+i (i=0..14)에 대응
- frontier 주변 semantic summary(`objs_list`)에 사용 (main_llm_zeroshot.py:741)

### 10.2 `category_to_id` (6개) — LLM scoring 대상 목표 카테고리
```python
category_to_id = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]
```
- `construct_dist(objs_list)`에서 이 6개 카테고리에 대해 LLM 점수 산출 (main_llm_zeroshot.py:478)
- `new_dist[category_to_id.index(cname)]`로 목표 카테고리의 점수 추출 (main_llm_zeroshot.py:754)
- 즉 **LLM scoring의 출력 차원은 6** (hm3d 15개가 아님)

> iGibson 이식 시:
> - hm3d_category(15개)는 semantic map 채널과 objs_list에 사용 → taxonomy adapter 대상
> - category_to_id(6개)는 탐색 목표 정의에 사용 → iGibson 목표 정의와 매핑 필요
> - 두 리스트는 역할이 다르므로 독립적으로 관리해야 함

---

## 11. remove_small_points의 Goal_score 반환값 (추가)

`remove_small_points` 함수 (main_llm_zeroshot.py:301-350)는 3개 값을 반환:
- `Goal_edge`: (H,W) binary edge mask
- `Goal_point`: (H,W) int label map (frontier centroids)
- **`Goal_score`**: list[float] — 각 frontier의 cost score (area + distance 기반)

`Goal_score`는 LLM scoring이 불가능할 때(objs_list 비어있음 또는 found_goal일 때)
fallback 점수로 사용됨:
```python
frontier_score_list[e].append(Goal_score[lay]/max(Goal_score) * 0.1 + 0.1)
```
(main_llm_zeroshot.py:756)

> iGibson 이식 시 `remove_small_points`는 main loop 내부 함수이므로 변경 불필요.
> 단, 함수가 `FMMPlanner`를 내부적으로 사용한다는 점에 주의 (main_llm_zeroshot.py:308).