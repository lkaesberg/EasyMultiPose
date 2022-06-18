import json
from pathlib import Path


def create_camera_file(cx, cy, depth_scale, fx, fy, height, width, home_path: Path):
    (home_path / "camera.json").write_text(
        json.dumps(
            {"cx": cx, "cy": cy, "depth_scale": depth_scale, "fx": fx, "fy": fy, "height": height, "width": width},
            indent=2))


if __name__ == '__main__':
    create_camera_file(641.068883438, 507.72159802, 0.1, 1075.65091572, 1073.90347929, 1024, 1280, Path("/home/lars"))
