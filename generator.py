import os
import xml.etree.ElementTree as ET
import mjfit

def generate_assembly_assets(
    output_dir="assets",
    peg_radius=0.015,         # 销轴半径 15mm
    clearance=0.0005,         # 单边间隙 0.5mm (孔半径 = peg_radius + clearance)
    hole_depth=0.06,          # 孔深 60mm
    friction=(0.3, 0.005, 0.0001), # (滑动摩擦, 扭转摩擦, 滚动摩擦)
    solref=(0.002, 1.0),      # 时间常数 2ms, 阻尼比 1.0
    solimp=(0.95, 0.99, 0.001, 0.5, 2),
):
    os.makedirs(output_dir, exist_ok=True)
    solref_str = f"{solref[0]} {solref[1]}"
    solimp_str = " ".join(map(str, solimp))
    fric_str = " ".join(map(str, friction))

    # 1. 生成孔座 (使用 mjfit 保证微米级间隙不被凸包破坏)
    hole_config = {
        "shape": "circle",
        "radius": peg_radius,
        "clearance": clearance,
        "height": hole_depth,
    }
    hole_mjcf_str = mjfit.build(hole_config, standalone=True).to_xml()
    hole_root = ET.fromstring(hole_mjcf_str)

    # 注入用户指定的软接触参数与摩擦属性
    for geom in hole_root.iter("geom"):
        geom.set("solref", solref_str)
        geom.set("solimp", solimp_str)
        geom.set("friction", fric_str)
        geom.set("condim", "4")

    hole_xml_path = os.path.join(output_dir, "hole.xml")
    ET.ElementTree(hole_root).write(hole_xml_path, encoding="utf-8", xml_declaration=True)

    # 2. 生成圆柱销 (原生 Cylinder Geom，保留分析级接触圆滑度)
    peg_xml_path = os.path.join(output_dir, "peg.xml")
    peg_xml_content = f"""<mujoco model="peg">
  <asset>
    <material name="peg_mat" specular="0.6" shininess="0.5" rgba="0.85 0.25 0.2 1" />
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom name="peg_geom" type="cylinder" size="{peg_radius} {hole_depth/2.0}" mass="0.1"
              material="peg_mat" condim="4" friction="{fric_str}"
              solref="{solref_str}" solimp="{solimp_str}" />
        <site name="peg_tip" pos="0 0 {-hole_depth/2.0}" size="0.002" rgba="0 1 0 1" />
      </body>
    </body>
  </worldbody>
</mujoco>"""
    with open(peg_xml_path, "w") as f:
        f.write(peg_xml_content)

    return peg_xml_path, hole_xml_path
