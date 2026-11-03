import numpy as np
from robosuite.controllers import load_controller_config
from generator import generate_assembly_assets
from franka_peg_in_hole_env import FrankaPegInHoleEnv

def main():
    # 1. 参数化更换零件：这里设定 0.4mm 间隙，中等柔顺程度
    clearance_eval = 0.0004
    peg_xml, hole_xml = generate_assembly_assets(
        clearance=clearance_eval,
        peg_radius=0.015,
        solref=(0.002, 1.0),     # 刚性较硬
        friction=(0.35, 0.005, 0.0001)
    )

    # 2. 机械臂笛卡尔阻抗控制器（OSC_POSE）
    controller_config = load_controller_config(default_controller="OSC_POSE")

    # 3. 实例化基准环境
    env = FrankaPegInHoleEnv(
        peg_xml_path=peg_xml,
        hole_xml_path=hole_xml,
        clearance=clearance_eval,
        controller_configs=controller_config,
        force_noise_std=0.25,
        control_freq=20,
    )

    print(">>> Franka Panda Peg-in-Hole 环境已就绪。")
    obs = env.reset()

    for step in range(500):
        # 控制策略输入仅有：obs['ee_pos'], obs['ee_quat'], obs['ee_vel'], obs['ee_force_noisy'], obs['ee_torque_noisy']
        # 发送末端下压并进行顺应性微幅搜索动作
        action = np.array([0.001 * np.sin(step * 0.2), 0.0, -0.02, 0.0, 0.0, 0.0])

        obs, reward, done, info = env.step(action)
        env.render()

        # 后台仅供标注与评估的【真值记录】（不可回传给 action 决策网络）
        if step % 25 == 0:
            gt = env.get_contact_ground_truth()
            print(f"--- Step {step} ---")
            print(f"  [控制端有限观测] 带噪力 Fz: {obs['ee_force_noisy'][2]:.2f} N | 末端高度 Z: {obs['ee_pos'][2]:.3f} m")
            print(f"  [评测端几何真值] 插入深度: {gt['insertion_depth']*1000:.1f} mm | 最大穿透: {gt['max_penetration']*1000:.3f} mm | 接触点数: {gt['num_active_contacts']}")

        if done:
            break

    env.close()

if __name__ == "__main__":
    main()
