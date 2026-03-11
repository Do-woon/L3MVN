try:
    from .habitat import construct_envs, construct_envs21
except ImportError:
    construct_envs = None
    construct_envs21 = None


def _count_non_floor_contacts(robot, scene) -> int:
    """Return the number of robot contacts excluding floor/self contacts."""
    import pybullet as p

    body_ids = robot.get_body_ids()
    if not body_ids:
        raise RuntimeError("Robot has no body ids; cannot evaluate collisions.")
    body_id = body_ids[0]

    floor_ids = set(getattr(scene, "floor_body_ids", []) or [])
    try:
        floor_z = float(scene.floor_heights[0])
    except (AttributeError, IndexError, TypeError):
        floor_z = 0.0

    p.performCollisionDetection()
    contacts = p.getContactPoints(bodyA=body_id)

    non_floor_contacts = 0
    for contact in contacts:
        body_b = contact[2]  # bodyB id
        if body_b == body_id or body_b in floor_ids:
            continue
        # Keep floor-like contacts filtered when floor body ids are incomplete.
        normal_z = contact[7][2]
        contact_z = contact[5][2]
        if abs(normal_z) > 0.7 and contact_z < floor_z + 0.15:
            continue
        non_floor_contacts += 1
    return non_floor_contacts


def _spawn_collision_free(
    robot,
    scene,
    sim,
    *,
    floor: int = 0,
    max_attempts: int = 20,
    settle_steps: int = 5,
    z_offset: float = 0.1,
) -> int:
    """Spawn robot at a traversable point with no non-floor contacts.

    Returns
    -------
    int
        1-based attempt index that produced a collision-free spawn.

    Raises
    ------
    RuntimeError
        If no collision-free spawn is found within ``max_attempts``.
    """
    import numpy as np

    start_orn = [0.0, 0.0, 0.0, 1.0]
    last_contact_count = None

    for attempt in range(1, max_attempts + 1):
        _, pos = scene.get_random_point(floor=floor)
        pos = np.array(pos, dtype=np.float32).copy()
        pos[2] += z_offset

        robot.set_position_orientation(pos, start_orn)
        robot.reset()
        robot.keep_still()
        for _ in range(settle_steps):
            sim.step()

        contact_count = _count_non_floor_contacts(robot, scene)
        if contact_count == 0:
            return attempt
        last_contact_count = contact_count

    raise RuntimeError(
        "Failed to find collision-free spawn after "
        f"{max_attempts} attempts (last non-floor contact count: {last_contact_count})."
    )


def make_vec_envs(args):
    if getattr(args, "use_igibson", 0):
        return _make_igibson_vec_envs(args)
    if args.task_config == "tasks/objectnav_gibson.yaml":
        envs = construct_envs(args)
    else:
        envs = construct_envs21(args)
    envs = VecPyTorch(envs, args.device)
    return envs


def _make_igibson_vec_envs(args):
    """Assemble the iGibson env stack and return a Stage-1 SingleEnvVecWrapper.

    Stack (bottom → top)
    --------------------
    Simulator + Scene + Robot
      → DiscreteActionExecutor
      → ObsAdapter
      → SemanticTaxonomy (class reference)
      → EnvWrapper          (Stage-1 obs: (5,H,W) numpy)
      → SingleEnvVecWrapper (Stage-1 batch: (1,5,H,W) numpy)

    Returns
    -------
    SingleEnvVecWrapper
        reset()  -> (obs (1,5,H,W), infos list[dict])
        plan_act_and_preprocess(planner_inputs)
                 -> (obs (1,5,H,W), fail_case list[dict], done ndarray, infos list[dict])
        close()  -> None
    """
    import igibson
    igibson.assets_path = getattr(
        args, "igibson_assets_path",
        "/mount/nas2/users/dukim/vla_ws/igibson/data/assets",
    )
    igibson.ig_dataset_path = getattr(
        args, "igibson_dataset_path",
        "/mount/nas2/users/dukim/vla_ws/igibson/data/ig_dataset",
    )
    igibson.key_path = getattr(
        args, "igibson_key_path",
        "/mount/nas2/users/dukim/vla_ws/igibson/data/igibson.key",
    )

    from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
    from igibson.robots.turtlebot import Turtlebot
    from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
    from igibson.simulator import Simulator
    from igibson.utils.semantics_utils import get_class_name_to_class_id

    from envs.igibson.discrete_action_executor import DiscreteActionExecutor
    from envs.igibson.env_wrapper import EnvWrapper
    from envs.igibson.obs_adapter import ObsAdapter
    from envs.igibson.semantic_taxonomy import SemanticTaxonomy
    from envs.igibson.vec_env_wrapper import SingleEnvVecWrapper

    frame_width  = getattr(args, "frame_width",        160)
    frame_height = getattr(args, "frame_height",       120)
    scene_name   = getattr(args, "igibson_scene",      "Rs_int")
    goal_name    = getattr(args, "goal_name",          "chair")
    goal_cat_id  = getattr(args, "goal_cat_id",        1)
    max_steps    = getattr(args, "max_episode_length", 500)

    # 1. Simulator
    settings = MeshRendererSettings(
        enable_shadow=False,
        enable_pbr=False,
        msaa=False,
        optimized=True,
        load_textures=True,
    )
    sim = Simulator(
        mode="gui_interactive",
        image_width=frame_width,
        image_height=frame_height,
        vertical_fov=45,
        physics_timestep=1 / 240.0,
        render_timestep=1 / 10.0,
        rendering_settings=settings,
    )

    scene = InteractiveIndoorScene(
        scene_name,
        trav_map_type="no_obj",
        trav_map_resolution=0.1,
        trav_map_erosion=2,
        build_graph=True,
    )
    sim.import_scene(scene)

    robot = Turtlebot(action_type="continuous")
    sim.import_object(robot)

    # Initial robot placement: retry until collision-free or raise.
    spawn_attempts = _spawn_collision_free(robot, scene, sim)
    sim._spawn_collision_free_attempts = spawn_attempts
    for _ in range(10):
        sim.step()

    # 2. class_id_to_name
    name_to_id = get_class_name_to_class_id()
    class_id_to_name = {v: k for k, v in name_to_id.items()}

    # 3. Stack layers
    executor    = DiscreteActionExecutor(robot=robot, scene=scene)
    obs_adapter = ObsAdapter()

    inner_wrapper = EnvWrapper(
        igibson_env=sim,
        robot=robot,
        scene=scene,
        action_executor=executor,
        obs_adapter=obs_adapter,
        semantic_taxonomy=SemanticTaxonomy,
        goal_name=goal_name,
        goal_cat_id=goal_cat_id,
        class_id_to_name=class_id_to_name,
        max_steps=max_steps,
        args=args,
    )

    return SingleEnvVecWrapper(inner_wrapper)


# Adapted from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class VecPyTorch():

    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.device = device

    def reset(self):
        obs, info = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def step(self, actions):
        actions = actions.cpu().numpy()
        obs, reward, done, info = self.venv.step(actions)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def get_rewards(self, inputs):
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(reward).float()
        return reward

    def plan_act_and_preprocess(self, inputs):
        obs, reward, done, info = self.venv.plan_act_and_preprocess(inputs)
        obs = torch.from_numpy(obs).float().to(self.device)
        # reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def close(self):
        return self.venv.close()
