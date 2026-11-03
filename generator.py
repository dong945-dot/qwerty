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

    # ==========================================
    # 1. 生成孔座 (hole.xml)
    # ==========================================
    hole_config = {
        "kind": "peg",  # 必须指定 kind，否则 mjfit 报错
        "hole": {
            "type": "circle",
            "radius": peg_radius + clearance,
        },
        "clearance": clearance,
        "plate_thickness": hole_depth,
    }
    
    # mjfit.build 返回的已经是 XML 字符串，不需要 .to_xml()
    hole_mjcf_str = mjfit.build(hole_config, standalone=True)
    
    # 为了防止 mjfit 返回的是完整场景，我们只提取 <worldbody> 下的核心物理结构
    # 构建一个符合 robosuite 规范的全新 XML 树
    hole_root = ET.Element("mujoco", model="hole")
    worldbody = ET.SubElement(hole_root, "worldbody")
    outer_body = ET.SubElement(worldbody, "body")
    object_body = ET.SubElement(outer_body, "body", name="object")

    # 解析 mjfit 生成的字符串
    parsed_root = ET.fromstring(hole_mjcf_str)
    
    # 寻找所有 geom 并注入到我们的 object_body 中
    for geom in parsed_root.iter("geom"):
        geom.set("solref", solref_str)
        geom.set("solimp", solimp_str)
        geom.set("friction", fric_str)
        geom.set("condim", "4")
        geom.set("group", "0") # 必须指定 group 属性，否则 robosuite 报错
        object_body.append(geom)

    hole_xml_path = os.path.join(output_dir, "hole.xml")
    ET.ElementTree(hole_root).write(hole_xml_path, encoding="utf-8", xml_declaration=True)

    # ==========================================
    # 2. 生成圆柱销 (peg.xml)
    # ==========================================
    peg_xml_path = os.path.join(output_dir, "peg.xml")
    
    # 注意：增加了 group="0"，以及必需的三个 site
    peg_xml_content = f"""<mujoco model="peg">
  <asset>
    <material name="peg_mat" specular="0.6" shininess="0.5" rgba="0.85 0.25 0.2 1" />
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom name="peg_geom" type="cylinder" size="{peg_radius} {hole_depth/2.0}" mass="0.1"
              material="peg_mat" condim="4" friction="{fric_str}"
              solref="{solref_str}" solimp="{solimp_str}" group="0" />
              
        <!-- robosuite 必需的基础 site，用于计算包围盒和接触点 -->
        <site name="bottom_site" pos="0 0 {-hole_depth/2.0}" size="0.001" rgba="0 0 0 0" />
        <site name="top_site" pos="0 0 {hole_depth/2.0}" size="0.001" rgba="0 0 0 0" />
        <site name="horizontal_radius_site" pos="{peg_radius} 0 0" size="0.001" rgba="0 0 0 0" />
        
        <!-- 你原本用来标记的 tip site -->
        <site name="peg_tip" pos="0 0 {-hole_depth/2.0}" size="0.002" rgba="0 1 0 1" />
      </body>
    </body>
  </worldbody>
</mujoco>"""
    
    with open(peg_xml_path, "w") as f:
        f.write(peg_xml_content)

    return peg_xml_path, hole_xml_path

if __name__ == "__main__":
    # 方便单独测试生成脚本
    peg, hole = generate_assembly_assets()
    print(f"生成成功: {peg}, {hole}")
