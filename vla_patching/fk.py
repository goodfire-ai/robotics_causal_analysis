"""Forward kinematics of predicted action chunks and the normalized trajectory shift E."""
from __future__ import annotations

import os

import numpy as np

from .data import DATA

ROBOT_XML = DATA / "robot" / "franka_droid_kinematics.xml"


class FK:
    """Grasp-site trajectory of a predicted action chunk on the DROID Franka arm.

    MolmoAct2 and MolmoBot predict absolute joint positions; pi0.5 predicts joint velocities,
    integrated from the current state (x0.105, clipped to [-1, 1]). Every trajectory in a
    comparison starts from the recipient's state.
    """

    def __init__(self, xml=ROBOT_XML):
        os.environ.setdefault("MUJOCO_GL", "disable")
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.indices = [self.model.joint(f"fr3_joint{i}").qposadr[0] for i in range(1, 8)]
        self.site = self.model.site("gripper/grasp_site").id

    def trajectory(self, action, model_name, state):
        a = np.asarray(action, dtype=np.float64)
        q = (np.asarray(state)[:7] + np.cumsum(np.clip(a[:, :7], -1, 1), axis=0) * .105
             if model_name == "pi05" else a[:, :7])
        points = []
        for pose in [state] + list(q):
            self.data.qpos[self.indices] = np.asarray(pose)[:7]
            self.mj.mj_forward(self.model, self.data)
            points.append(self.data.site_xpos[self.site].copy())
        return np.round(np.asarray(points), 5)


def projection(delta, axis, threshold):
    """E = <delta, axis> / ||axis||^2, or None when the reference contrast is below `threshold` RMS."""
    den = float(np.sum(axis ** 2))
    rms = float(np.sqrt(np.mean(np.sum(axis ** 2, axis=-1))))
    num = float(np.sum(delta * axis))
    return (num / den if den > 1e-12 and rms >= threshold else None), num, den, rms
