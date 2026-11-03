import os
from robosuite.models.objects import MujocoXMLObject

class DynamicPegObject(MujocoXMLObject):
    def __init__(self, xml_path, name="peg"):
        super().__init__(
            fname=os.path.abspath(xml_path),
            name=name,
            joints=[{"type": "free", "damping": "0.0005"}],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

class DynamicHoleObject(MujocoXMLObject):
    def __init__(self, xml_path, name="hole"):
        super().__init__(
            fname=os.path.abspath(xml_path),
            name=name,
            joints=None,  # 静态锚定在桌面上
            obj_type="all",
            duplicate_collision_geoms=False,
        )
