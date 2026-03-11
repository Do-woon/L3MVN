# L3MVN → iGibson Porting Notes (Interface Inventory) — v1.3

> 업데이트 내용(v1):
> - TODO#1: planner_inputs 필수 키/shape/type 확정
> - TODO#2: infos['sensor_pose'] 포맷/단위 확정
> - TODO#3: frontier 후보 표현/스코어링 데이터 구조 확정
>
> 업데이트 내용(v1.1 — 코드 대조 검증 반영):
> - infos 필수 키 전체 나열 (sensor_pose 외 eve_angle, goal_cat_id, goal_name, clear_flag 추가)
> - Semantic_Mapping 반환값 4개 명시
> - remove_small_points의 Goal_score 반환값 문서화
> - construct_dist의 category_to_id(6개) vs hm3d_category(15개) 구분 명시
>
> 업데이트 내용(v1.2 — 최종 규약 통일):
> - EnvWrapper/SingleEnvVecWrapper 반환 계약을 Stage 1(5ch raw)으로 고정
> - Stage 2(20ch)는 `Sem_Exp_Env_Agent._preprocess_obs()` 책임으로 명시
> - Habitat 원본과 동일한 책임 분리를 iGibson 포팅에서도 유지한다고 명시
>
> 업데이트 내용(v1.3 — planner_inputs/action 규약 재정렬):
> - Habitat 경로를 canonical contract로 고정: `planner_inputs`는 8-key 입력 계약 사용
> - `action`은 `planner_inputs`의 필수 입력 키가 아니라 env 내부 planning 결과라는 점 명시
> - iGibson EnvWrapper도 동일하게 8-key를 받아 내부에서 action을 계산하도록 정렬
>
> 근거 소스:
> - planner_inputs 구성: main_llm_zeroshot.py:842-857
> - sensor_pose 소비: main_llm_zeroshot.py:594-597
> - sensor_pose 생성: envs/habitat/objectgoal_env.py:396-397
> - frontier 후보 생성/표현: main_llm_zeroshot.py:684-756

---

## 1. Env call contract (최종 규약)

### 1.1 iGibson adapter 반환 계약 (Stage 1)
- `make_vec_envs(args)`의 iGibson 분기는 **Stage 1 raw obs를 반환하는 env object**를 반환한다.
- `EnvWrapper.reset()` / `EnvWrapper.plan_act_and_preprocess()`의 obs는 Stage 1 raw obs이다.
- `SingleEnvVecWrapper`는 batch 차원만 추가하며, obs는 Stage 1 batch obs이다.
- Stage 1 obs 채널/shape:
  - single env: `(5, H, W)` = RGB(3) + Depth(1) + SemanticID(1)
  - single-env batch wrapper: `(1, 5, H, W)`
- Stage 1에서의 데이터 의미:
  - depth: raw metres
  - semantic: semantic id single-channel
  - one-hot 생성/centimeter 변환은 여기서 수행하지 않는다.
- Habitat 원본에서도 env는 raw obs를 반환하고, Stage 2는 `Sem_Exp_Env_Agent._preprocess_obs()`가 생성한다.
- iGibson 포팅에서도 같은 책임 분리를 유지한다.

### 1.2 L3MVN preprocessing 이후 계약 (Stage 2)
- `Sem_Exp_Env_Agent._preprocess_obs()`가 Stage 1을 받아 Stage 2를 생성한다.
- Stage 2 obs 채널: RGB(3) + Depth(1) + Semantic one-hot(16) = 20ch
- Stage 2에서 수행되는 전처리:
  - depth 전처리 및 cm 변환
  - semantic id -> 16채널 one-hot 변환
- `Semantic_Mapping` 입력 계약 `(N, 20, H, W)`는 adapter 출력이 아니라 `_preprocess_obs()` 결과 기준이다.

### 1.3 plan_act_and_preprocess() 반환값
- `obs, fail_case, done, infos = envs.plan_act_and_preprocess(planner_inputs)` (main_llm_zeroshot.py:860)
- reward 대신 **fail_case**가 반환됨(zeroshot 루프)
  - `fail_case`: dict `{'collision': int, 'success': int, 'detection': int, 'exploration': int}` (agents/sem_exp.py:76-80)
  - VecPyTorch에서 torch 변환 없이 pass-through됨 (envs/__init__.py:56 — 주석 처리된 `# reward = torch.from_numpy(reward).float()`)
- done: ndarray/bool list (env별 episode 종료)

---

## 2. planner_inputs spec (TODO#1 완료)

main_llm_zeroshot.py에서 env 호출 직전에 planner_inputs를 env 개수만큼 dict로 채움. :contentReference[oaicite:5]{index=5}

각 env index e에 대해:

### 2.1 필수 키 (항상 설정)
- `map_pred`: `local_map[e, 0, :, :]` → numpy 2D (H,W)
- `exp_pred`: `local_map[e, 1, :, :]` → numpy 2D (H,W)
- `pose_pred`: `planner_pose_inputs[e]` → numpy 1D (7,)
  - [0:3] = agent (x,y,o?) 연속좌표 + origins 적용 후
  - [3:7] = local map boundaries (lmb) (gx1,gx2,gy1,gy2)
- `goal`: `local_goal_maps[e]` → numpy 2D (H,W) (binary goal mask)
- `map_target`: `target_point_map[e]` → numpy 2D (H,W) (frontier label map: 0=none, 1..N=frontier index)
- `new_goal`: bool, `l_step == args.num_local_steps - 1`
- `found_goal`: int/bool (0/1), 목표 감지 플래그
- `wait`: bool/int, `wait_env[e] or finished[e]`

### 2.2 조건부 키 (visualize 시만)
- `map_edge`: `target_edge_map[e]` → numpy 2D (H,W) (frontier edge mask)
- `sem_map_pred`: `local_map[e, 4:, :, :].argmax(0)` → numpy 2D (H,W), semantic argmax id map

### 2.3 shape 요약
- `map_pred, exp_pred, goal, map_target, map_edge, sem_map_pred` : (local_w, local_h) = (H,W)
- `pose_pred`: (7,)

> iGibson 어댑터에서 반드시 맞춰야 할 것은 **planner_inputs의 key set과 각 타입/shape**임.
> 특히 `pose_pred`의 7D 규약은 L3MVN 내부 로컬/글로벌 좌표 변환과 직결됨.

### 2.4 action 처리 규약 (Habitat 기준)
- canonical 실행 경로에서 env 입력은 8-key `planner_inputs`이며, `action`은 입력 필수 키가 아니다.
- action은 env 내부 local planner(`_plan` 등)가 `planner_inputs`를 사용해 계산한다.
- iGibson 포팅도 동일 규약을 목표로 하며, `planner_inputs["action"]` 직접 강제는 임시 디버그 경로로만 취급한다.
- 최종 목표:
  - main loop: 8-key planner_inputs 생성/전달
  - EnvWrapper: 8-key 소비 -> action 계산 -> simulator 실행 -> `(obs, fail_case, done, info)` 반환

---

## 3. infos 필수 키 전체 spec (TODO#2 완료 + v1.1 보강)

### 3.1 sensor_pose 소비(사용) 위치
- main_llm_zeroshot.py:594-597:
  - `poses = torch.from_numpy([infos[i]['sensor_pose'] ...])` 로 묶어 sem_map_module에 전달
- 즉 infos['sensor_pose']는 "절대 pose"가 아니라 **map integration에 쓰는 ego-motion**으로 사용됨.

### 3.2 sensor_pose 생성(정의) 위치
- envs/habitat/objectgoal_env.py:396-397 (ObjectGoal_Env.step):
  - `dx, dy, do = self.get_pose_change()`
  - `self.info['sensor_pose'] = [dx, dy, do]`
- envs/habitat/objectgoal_env21.py:192-193 (ObjectGoal_Env21.step) — 동일 구조

### 3.3 get_pose_change()의 의미
- `get_pose_change()`는 (objectgoal_env.py:541-548):
  - `curr_sim_pose = self.get_sim_location()` (x,y,o)
  - `pu.get_rel_pose_change(curr_sim_pose, last_sim_location)` 로 상대변화를 계산
- `pu.get_rel_pose_change` (envs/utils/pose.py:11-21):
  - dx, dy는 **ego-frame** 기준 상대 이동 (에이전트 heading o1 기준 회전된 좌표)
  - do = o2 - o1 (yaw 변화)

따라서:
- `sensor_pose = [dx, dy, do]`는 **직전 step 대비 상대 이동량(ego-motion)**
- `dx, dy`는 **ego-frame에서의 거리 변화(미터 단위)**
  (Habitat sim 위치가 meters이며, get_rel_pose_change가 L2 distance에 ego-frame 각도를 적용)
- `do`는 `get_sim_location()`의 o가 라디안 범위로 정규화되므로, **yaw 변화(라디안)**

### 3.4 infos의 기타 필수 키 (v1.1 추가)

main_llm_zeroshot.py에서 infos에서 직접 참조하는 **모든** 키:

| 키 | 타입 | 설정 위치 | 소비 위치 | 용도 |
|---|---|---|---|---|
| `sensor_pose` | list[float] (3,) | env.step | main:594-597 | sem_map_module 입력 |
| `eve_angle` | int | sem_exp.reset:110, plan_act:191 | main:599-601 | sem_map_module 5번째 인자 (elevation) |
| `goal_cat_id` | int | env.reset | main:728, 827 | 목표 semantic 채널 index (cn = goal_cat_id + 4) |
| `goal_name` | str | env.reset | main:729, 579 | LLM scoring 대상, 로깅 |
| `clear_flag` | int (0/1) | sem_exp._plan:154,163 | main:672 | 맵 초기화 트리거 |
| `spl` | float | env.step (done 시) | main:576 | 평가 메트릭 |
| `success` | int (0/1) | env.step (done 시) | main:577 | 평가 메트릭 |
| `distance_to_goal` | float | env.step (done 시) | main:578 | 평가 메트릭 |
| `g_reward` | int (0/1) | sem_exp.plan_act:151 | main:813 | (조건부) Gibson task에서만 사용 |

> iGibson 이식 시 핵심:
> - infos['sensor_pose']는 "world pose"가 아니라 "상대 delta pose"로 제공해야 함.
> - dx/dy/do는 **각 step의 macro-action 실행 결과**로 계산해야 L3MVN mapper가 정상 동작.
> - macro-action 구현 방식은 자유이며, v1에서는 base pose teleport + collision check 방식도 허용된다.
>   (중요한 것은 저수준 제어 방식이 아니라, 최종 실행 결과 delta와 충돌 처리 일관성이다.)
> - **eve_angle**은 카메라 elevation 상태로, Look Up/Down 액션(4,5)과 연동됨.
>   iGibson adapter에서 내부 상태로 관리하고 infos에 매 step 반영해야 함.
> - **clear_flag**는 plan_act_and_preprocess 내부에서 설정되므로, adapter의
>   plan_act_and_preprocess 구현에서 collision 감지/replan 로직을 포함해야 함.

---

## 4. Frontier candidate representation (TODO#3 완료)

zeroshot 구현은 “frontier”를 별도 객체로 들고 가지 않고,
`target_edge_map` / `target_point_map`이라는 2D map으로 표현함. :contentReference[oaicite:11]{index=11}

### 4.1 frontier 생성 파이프라인(요약)
- `local_ob_map[e]` : obstacle map (local_map ch0) dilation
- `local_ex_map[e]` : explored/free-like mask를 contour로 채움
- `target_edge = local_ex_map - local_ob_map` 후 threshold → edge mask
- `remove_small_points(ob_map, target_edge, threshold, pose)` 호출 (main_llm_zeroshot.py:301-350):
  - connected components를 찾고
  - 조건(area > threshold, 50 < dist < 500)으로 필터링
  - cost = area + dist_bonus로 정렬 (내림차순), 최대 4개까지 선택 (i == 3에서 break)
  - 각 component에 대해:
    - `Goal_edge` : edge mask (binary)
    - `Goal_point` : centroid를 frontier index (1..K)로 라벨링
    - **`Goal_score`** : list[float] — 각 frontier의 cost 점수

**반환값 3개**: `Goal_edge, Goal_point, Goal_score` (main_llm_zeroshot.py:350)

`Goal_score`의 사용:
- LLM scoring 불가 시(objs_list 비어있음 또는 found_goal) fallback 점수로 사용:
  `frontier_score_list[e].append(Goal_score[lay]/max(Goal_score) * 0.1 + 0.1)` (main_llm_zeroshot.py:756)

즉:
- `target_edge_map[e]` : (H,W) binary
- `target_point_map[e]` : (H,W) int labels
  - 0: none
  - 1..K: frontier id (ranking/selection용, 최대 K=4)

### 4.2 frontier “local window” 표현(fmb)
각 frontier id (lay=0..K-1)에 대해:
- `f_pos = argwhere(target_point_map == lay+1)`로 centroid 픽셀 좌표 획득
- `fmb = get_frontier_boundaries((f_pos[0][0], f_pos[0][1]), (local_w/6, local_h/6), (local_w, local_h))`
  - 결과: `[gx1,gx2,gy1,gy2]` 형태의 bounding box :contentReference[oaicite:13]{index=13}

### 4.3 frontier 주변 semantic summary(objs_list)
- fmb 영역 내에서 semantic 채널( local_map[e][4+se_cn] )의 합이 0이 아니면 해당 카테고리를 포함:
  - `objs_list.append(hm3d_category[se_cn])` :contentReference[oaicite:14]{index=14}
- 즉 frontier 주변 요약은:
  - `frontier_id`
  - `frontier_center_pixel`
  - `frontier_bbox(fmb)`
  - `objs_list: list[str]` (semantic class names)

### 4.4 zero-shot scoring 출력(실제 코드 경로)
현재 zeroshot 구현은 `construct_dist(objs_list)` → softmax 후
`new_dist[category_to_id.index(cname)]`를 frontier 점수로 사용 (main_llm_zeroshot.py:744-754)

**중요: `category_to_id`와 `hm3d_category`는 서로 다른 리스트**

- `hm3d_category` (15개): semantic map 채널 매핑에 사용. `objs_list`는 이 리스트에서 가져옴.
  ```python
  objs_list.append(hm3d_category[se_cn])  # main_llm_zeroshot.py:741
  ```
- `category_to_id` (6개): LLM scoring 대상 목표 카테고리.
  ```python
  category_to_id = [“chair”, “bed”, “plant”, “toilet”, “tv_monitor”, “sofa”]
  ```
  `construct_dist(objs_list)`는 이 6개에 대해서만 LLM 점수를 산출 (main_llm_zeroshot.py:478):
  ```python
  for label in category_to_id:
      TEMP_STR = query_str + “ “ + label + “.”
      score = scoring_fxn(TEMP_STR)
      TEMP.append(score)
  ```
  따라서 **LLM scoring 출력 차원은 6** (hm3d 15개가 아님).

> iGibson 이식 시:
> - frontier 자체 표현은 “label map(2D)”로 유지하면 가장 이식이 쉬움.
> - 핵심 갭은 semantic taxonomy( hm3d_category vs iGibson categories )라서,
>   objs_list 생성부만 curated taxonomy adapter로 교체하면 prompt/scoring 루프를 유지 가능.
> - `category_to_id`(6개)는 목표 카테고리 정의로, iGibson에서 지원하는 목표 객체 범위에 따라
>   별도 매핑이 필요할 수 있음.

---

## 5. Semantic_Mapping 모듈 상세 계약 (v1.1 추가)

### 5.1 forward() 시그니처 및 반환값
```python
# model.py:63
def forward(self, obs, pose_obs, maps_last, poses_last, eve_angle):
```

**인자:**
- `obs`: (N, C, H, W) where C=20. ch3=depth(cm), ch4..19=semantic one-hot(16ch)
  - 이 `obs`는 iGibson adapter raw obs가 아니라, `Sem_Exp_Env_Agent._preprocess_obs()` 이후 Stage 2 텐서임.
- `pose_obs`: (N, 3) — infos['sensor_pose']에서 추출한 [dx, dy, do]
- `maps_last`: (N, nc, local_w, local_h) — 이전 local_map (nc=20)
- `poses_last`: (N, 3) — 이전 local_pose [x, y, o]
- `eve_angle`: (N,) ndarray — infos['eve_angle']에서 추출

**반환값 (model.py:229):**
```python
return translated, map_pred, map_pred_stair, current_poses
```
main loop에서의 변수명 (main_llm_zeroshot.py:517-518):
```python
increase_local_map, local_map, local_map_stair, local_pose = \
    sem_map_module(obs, poses, local_map, local_pose, eve_angle)
```
- `increase_local_map` (=translated): agent view를 global frame으로 변환한 단일 프레임 맵
- `local_map` (=map_pred): 누적된 local map (max-pooling으로 병합)
- `local_map_stair` (=map_pred_stair): 계단 감지용 별도 맵 (main_llm_zeroshot.py:629-630에서 사용)
- `local_pose` (=current_poses): 업데이트된 pose [x, y, o]

### 5.2 pose 업데이트 내부 로직 (model.py:140-155)
`get_new_pose_batch`에서 pose_obs(=sensor_pose)를 현재 pose에 적용:
- dx, dy는 **ego-frame**에서의 이동량으로, sin/cos(현재 heading)를 적용해 global frame으로 변환
- do(라디안)는 `57.29577951308232`(=180/pi)으로 곱해져 **degree**로 변환 후 pose에 합산
- 즉 Semantic_Mapping 내부에서 pose는 **degree** 단위로 관리됨

---

## 6. Semantic taxonomy 상태

### 6.1 iGibson GT semantic remap (curated)
- L3MVN은 hm3d_category(15개)의 정해진 semantic class 문자열을 사용한다.
- iGibson semantic id -> class name -> L3MVN id remap은
  `SemanticTaxonomy`의 curated alias table로 deterministic하게 처리한다.
- Stage 1에서는 remapped semantic id single-channel만 전달하고,
  Stage 2 one-hot(16ch)은 `_preprocess_obs()`가 생성한다.
- 애매한 클래스(`cabinet`, `bookshelf`, `tv_stand` 등)는 0(background/ignore)로 유지한다.

## 7. 기타 note

- Look Up/Down은 Sem_Exp_Env_Agent._plan()이 실제로 반환하는 유효 action이며, L3MVN의 sem_exp 실행 계약은 사실상 discrete(6)이다.
- objectgoal_env.py의 Discrete(3)/주석은 현재 sem_exp 실행 경로의 기준이 아니므로, 포팅 시 action contract 근거로 사용하지 않는다.
