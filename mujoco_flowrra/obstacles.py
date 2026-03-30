"""
obstacles.py

Reads static and moving obstacles directly from MuJoCo's C++ arrays.
"""

from typing import Any, Dict, List

import numpy as np


class ObstacleManager:
    def __init__(self, model: Any, data: Any):
        self.model = model
        self.data = data

        # 1. Find Static Geometries (Pillars)
        self.static_geom_ids = []
        for i in range(self.model.ngeom):
            name = (
                self.model.names[self.model.name_geomadr[i] :]
                .split(b"\x00")[0]
                .decode("utf-8")
            )
            if name.startswith("obs_"):
                self.static_geom_ids.append(i)

        # 2. Find Moving Bodies (Spheres)
        self.moving_body_ids = []
        for i in range(self.model.nbody):
            name = (
                self.model.names[self.model.name_bodyadr[i] :]
                .split(b"\x00")[0]
                .decode("utf-8")
            )
            if name.startswith("mov_obs_"):
                self.moving_body_ids.append(i)

    def get_all_states(self) -> List[Dict[str, Any]]:
        states = []

        # -- Read Static Pillars --
        for geom_id in self.static_geom_ids:
            pos = self.data.geom_xpos[geom_id].copy()
            states.append(
                {
                    "id": f"static_{geom_id}",
                    "pos": pos,
                    "velocity": np.zeros(3),
                    "type": "static",
                }
            )

        # -- Read Moving Spheres --
        for body_id in self.moving_body_ids:
            pos = self.data.xpos[body_id].copy()

            # data.cvel stores 6D velocity: [rot_x, rot_y, rot_z, lin_x, lin_y, lin_z]
            # We only want the linear velocity [3:6]
            vel = self.data.cvel[body_id][3:6].copy()

            states.append(
                {
                    "id": f"moving_{body_id}",
                    "pos": pos,
                    "velocity": vel,
                    "type": "moving",
                }
            )

        return states

    def check_line_intersection(self, p1: np.ndarray, p2: np.ndarray):
        """Handled natively by GMM Affordance Field"""
        return False, []
