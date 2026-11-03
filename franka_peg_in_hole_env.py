import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat
import robosuite.utils.transform_utils as T

from assembly_objects import DynamicPegObject, DynamicHoleObject

class FrankaPegInHoleEnv(ManipulationEnv):
    def __init__(
        self,
        peg_xml_path,
        hole_xml_path,
        clearance=0.0005,
        robots="Panda",
        control_freq=20,
        horizon=500,
        controller_configs=None,
        force_noise_std=0.3,       # 模拟 F/T 传感器力噪声 (N)
        torque_noise_std=0.03,     # 模拟 F/T 传感器力矩噪声 (Nm)
        **kwargs,
    ):
        self.peg_xml_path = peg_xml_path
        self.hole_xml_path = hole_xml_path
        self.clearance = clearance
        self.force_noise_std = force_noise_std
        self.torque_noise_std = torque_noise_std

        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_offset = np.array([0, 0, 0.8])

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=control_freq,
            horizon=horizon,
            **kwargs,
        )

    def _load_model(self):
        super()._load_model()

        # 1. 机械臂初始构型（适合垂直向下插孔）
        self.robots[0].init_qpos = np.array([0, -0.2, 0, -2.1, 0, 1.9, 0.785])

        # 2. 桌台
        self.arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
        )
        self.arena.set_origin([0.15, 0, 0])

        # 3. 动态零件
        self.peg = DynamicPegObject(self.peg_xml_path, name="peg")
        self.hole = DynamicHoleObject(self.hole_xml_path, name="hole")

        self.model = ManipulationTask(
            mujoco_arena=self.arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.hole, self.peg],
        )

    def _setup_references(self):
        super()._setup_references()
        self.peg_body_id = self.sim.model.body_name2id(self.peg.root_body)
        self.hole_body_id = self.sim.model.body_name2id(self.hole.root_body)
        self.peg_tip_site_id = self.sim.model.site_name2id("peg_tip")

    def _setup_observables(self):
        """【控制端可见的有限观测（POMDP）】"""
        observables = super()._setup_observables()

        @sensor(modality="proprio")
        def ee_pos(obs_cache):
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])

        @sensor(modality="proprio")
        def ee_quat(obs_cache):
            return convert_quat(np.array(self.sim.data.site_xmat[self.robots[0].eef_site_id]), to="xyzw")

        @sensor(modality="proprio")
        def ee_vel(obs_cache):
            # 6维末端 Twist 速度 (前3位线速度，后3位角速度)
            lin_vel = np.array(self.sim.data.get_site_xvelp(self.robots[0].eef_site_name))
            ang_vel = np.array(self.sim.data.get_site_xvelr(self.robots[0].eef_site_name))
            return np.concatenate([lin_vel, ang_vel])

        @sensor(modality="proprio")
        def ee_force_noisy(obs_cache):
            raw_force = np.array(self.robots[0].ee_force)
            noise = np.random.normal(0, self.force_noise_std, size=3)
            return raw_force + noise

        @sensor(modality="proprio")
        def ee_torque_noisy(obs_cache):
            raw_torque = np.array(self.robots[0].ee_torque)
            noise = np.random.normal(0, self.torque_noise_std, size=3)
            return raw_torque + noise

        names = ["ee_pos", "ee_quat", "ee_vel", "ee_force_noisy", "ee_torque_noisy"]
        sensors = [ee_pos, ee_quat, ee_vel, ee_force_noisy, ee_torque_noisy]

        for s, name in zip(sensors, names):
            observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)

        return observables

    def _reset_internal(self):
        super()._reset_internal()

        # ==================== 工况变化 (Domain Randomization) ====================
        # 1. 孔座初始位置与航向偏差
        base_hole_pos = self.table_offset + np.array([0.45, 0.0, 0.03])
        pos_offset = np.random.uniform([-0.015, -0.015, 0.0], [0.015, 0.015, 0.0]) # 平移偏差 ±1.5cm
        yaw_offset = np.random.uniform(-np.deg2rad(15), np.deg2rad(15))            # 旋转偏差 ±15°

        hole_rot = T.euler2mat([0, 0, yaw_offset])
        self.sim.model.body_pos[self.hole_body_id] = base_hole_pos + pos_offset
        self.sim.model.body_quat[self.hole_body_id] = T.mat2quat(hole_rot)

        # 2. 将圆柱销初始化在机械臂末端稍上方进行初始对齐测试
        peg_pos = base_hole_pos + pos_offset + np.array([0, 0, 0.10])
        # 初始销姿态叠加微小倾角 (±2°)
        tilt_rot = T.euler2mat([np.random.uniform(-0.03, 0.03), np.random.uniform(-0.03, 0.03), yaw_offset])
        self.sim.data.set_joint_qpos(
            self.peg.joints[0],
            np.concatenate([peg_pos, T.mat2quat(tilt_rot)])
        )

        # 3. 接触刚度（柔顺性）与摩擦系数的在线微扰
        # 注：通过修改 geom_solref 调整时间常数（柔顺度），不直接使用弹簧刚度 k
        dyn_timeconst = np.random.uniform(0.0015, 0.0035) # 刚度范围扰动
        dyn_friction = np.random.uniform(0.2, 0.5)        # 摩擦力范围扰动

        peg_gid = self.sim.model.geom_name2id("peg_geom")
        self.sim.model.geom_solref[peg_gid][0] = dyn_timeconst
        self.sim.model.geom_friction[peg_gid][0] = dyn_friction

        self.sim.forward()

    def get_contact_ground_truth(self):
        """【仿真接触几何真值（仅供评估与标注，控制器严禁读取）】"""
        peg_tip = self.sim.data.site_xpos[self.peg_tip_site_id]
        hole_pos = self.sim.data.xpos[self.hole_body_id]
        
        # 1. 相对轴向插入深度与径向未对准量
        insertion_depth = hole_pos[2] - peg_tip[2]
        radial_misalignment = np.linalg.norm(peg_tip[:2] - hole_pos[:2])

        # 2. 微观接触点几何、穿透深度与真实接触反力
        contact_points = []
        max_penetration = 0.0
        normal_forces = []

        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            # 过滤出包含 peg 的接触对
            if c.geom1 == self.sim.model.geom_name2id("peg_geom") or c.geom2 == self.sim.model.geom_name2id("peg_geom"):
                if c.dist < 0:
                    penetration = -c.dist
                    max_penetration = max(max_penetration, penetration)
                
                # 获取接触点的无噪声解析约束力
                c_force = np.zeros(6)
                self.sim.data.c_force = c_force # placeholder
                normal_forces.append(abs(c.dist))

        return {
            "insertion_depth": float(insertion_depth),
            "radial_misalignment": float(radial_misalignment),
            "max_penetration": float(max_penetration),
            "num_active_contacts": len(normal_forces),
            "clearance_gt": float(self.clearance),
        }

    def reward(self, action=None):
        gt = self.get_contact_ground_truth()
        # 奖励设计：鼓励深入，惩罚偏心与过度穿透（物理卡死风险）
        reward = gt["insertion_depth"] * 10.0 - gt["radial_misalignment"] * 5.0
        if gt["max_penetration"] > 0.002: # 穿透超过 2mm 属于严重卡死倾向
            reward -= 2.0
        return reward
