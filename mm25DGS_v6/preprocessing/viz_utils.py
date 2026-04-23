import os
import numpy as np
import open3d as o3d
from open3d.visualization import webrtc_server


def visualize_geometries(geoms: list,
                         *,
                         remote_viz: bool,
                         title: str = "Visualization"):
    if remote_viz:
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        if "DISPLAY" in os.environ:
            os.environ.pop("DISPLAY")

        def _pick_port(requested: int) -> int:
            import socket
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", requested))
                s.close()
                return requested
            except OSError:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                    s2.bind(("127.0.0.1", 0))
                    return s2.getsockname()[1]

        port = _pick_port(8888)
        os.environ["WEBRTC_IP"] = "127.0.0.1"
        os.environ["WEBRTC_PORT"] = str(port)
        webrtc_server.enable_webrtc()

        print("=" * 80)
        print("Open3D Web Visualizer is running.")
        print("Create an SSH tunnel from your local machine:")
        print(f"  ssh -L {port}:localhost:{port} user@128.84.85.120")
        print(f"Then open: http://localhost:{port}")
        print("=" * 80)

        o3d.visualization.draw(
            geoms,
            title=f"{title} (Web Visualizer)",
            show_ui=True,
            width=1280,
            height=800,
            bg_color=(1.0, 1.0, 1.0, 1.0),
        )
    else:
        o3d.visualization.draw(
            geoms,
            title=title,
            show_ui=True,
            width=1280,
            height=800,
            bg_color=(1.0, 1.0, 1.0, 1.0),
        )


