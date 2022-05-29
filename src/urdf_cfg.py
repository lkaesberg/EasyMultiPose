class UrdfCFG:
    URDF_PATH = None


urdf_cfg = UrdfCFG()


def get_urdf_path():
    return urdf_cfg.URDF_PATH


def set_urdf_path(path):
    urdf_cfg.URDF_PATH = path
