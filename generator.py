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
        "kind": "peg", 
        "hole": {
            "type": "circle",
            "radius": peg_radius + clearance,
        },
        "clearance": clearance,
        "plate_thickness": hole_depth,
    }
    
    # 获取 mjfit 生成的原始 XML 字符串
    hole_mjcf_str = mjfit.build(hole_config, standalone=True)
    parsed_root = ET.fromstring(hole_mjcf_str)

    # 创建全新的、绝对符合 robosuite 规范的 XML 树
    hole_root = ET.Element("mujoco", model="hole")
    ET.SubElement(hole_root, "asset")
    worldbody = ET.SubElement(hole_root, "worldbody")
    outer_body = ET.SubElement(worldbody, "body")
    object_body = ET.SubElement(outer_body, "body", name="object")

    # 深度清洗：遍历 mjfit 里的所有 geom
    for raw_geom in parsed_root.iter("geom"):
        # 1. 如果带有 class 属性，直接删除（这是引发 KeyError 的根源）
        if "class" in raw_geom.attrib:
            del raw_geom.attrib["class"]
            
        # 2. 注入用户指定的物理参数
        raw_geom.set("solref", solref_str)
        raw_geom.set("solimp", solimp_str)
        raw_geom.set("friction", fric_str)
        raw_geom.set("condim", "4")
        raw_geom.set("group", "0") # 强制指定 group，避免 int(None) 错误
        
        # 3. 将这个干净的 geom 节点挂载到我们新建的 object_body 上
        object_body.append(raw_geom)
        
    # 补全 robosuite 强制需要的三个基础 site
    ET.SubElement(object_body, "site", name="bottom_site", pos=f"0 0 {-hole_depth/2.0}", size="0.001", rgba="0 0 0 0")
    ET.SubElement(object_body, "site", name="top_site", pos=f"0 0 {hole_depth/2.0}", size="0.001", rgba="0 0 0 0")
    ET.SubElement(object_body, "site", name="horizontal_radius_site", pos=f"{peg_radius+clearance} 0 0", size="0.001", rgba="0 0 0 0")

    hole_xml_path = os.path.join(output_dir, "hole.xml")
    ET.ElementTree(hole_root).write(hole_xml_path, encoding="utf-8", xml_declaration=True)

    # ==========================================
    # 2. 生成圆柱销 (peg.xml)
    # ==========================================
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
              solref="{solref_str}" solimp="{solimp_str}" group="0" />
              
        <!-- robosuite 必需的基础 site -->
        <site name="bottom_site" pos="0 0 {-hole_depth/2.0}" size="0.001" rgba="0 0 0 0" />
        <site name="top_site" pos="0 0 {hole_depth/2.0}" size="0.001" rgba="0 0 0 0" />
        <site name="horizontal_radius_site" pos="{peg_radius} 0 0" size="0.001" rgba="0 0 0 0" />
        
        <site name="peg_tip" pos="0 0 {-hole_depth/2.0}" size="0.002" rgba="0 1 0 1" />
      </body>
    </body>
  </worldbody>
</mujoco>"""
    
    with open(peg_xml_path, "w") as f:
        f.write(peg_xml_content)

    return peg_xml_path, hole_xml_path

if __name__ == "__main__":
    peg, hole = generate_assembly_assets()
    print(f"生成成功！\n销轴: {peg}\n孔座: {hole}")
