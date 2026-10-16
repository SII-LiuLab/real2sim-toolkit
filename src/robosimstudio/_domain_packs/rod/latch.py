# SPDX-License-Identifier: Apache-2.0
"""The kinematic grasp latch, and the standalone form a repro bundle replays.

A compliant parallel-jaw friction grasp in this contact stack cannot reliably
hold a dragged rod: the fingers still close and still push, but the *hold* is
this coupling.

**Import discipline: newton, warp, numpy, nothing else, and this module names no
package of its own.** It runs where only newton is installed, and the exporter's
leak scan reads this text -- so a cross-reference written here would fail the
scan rather than merely mislead.
"""

from __future__ import annotations

import warp as wp

__all__ = ["build_hooks", "hold_segment"]


@wp.kernel
def hold_segment(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    seg_idx_arr: wp.array(dtype=int),
    grip_idx_arr: wp.array(dtype=int),
    rel_tf_arr: wp.array(dtype=wp.transform),
    enabled: wp.array(dtype=int),
    gains: wp.array(dtype=float),
):
    """Pin the grasped segment at its taken pose in the gripper frame.

    Position-only: the segment may swing about the pinch but not translate
    out of it.

    The force is clamped to ``gains[2]``, so a contact fight makes the latch
    slip. Unclamped, the closed pads drive their proxy contacts into the pinned
    segment (~80 N squeeze reaction through the 20x proxies), and on ``weave3``
    the rod left the bench at 72 m/s mid-carry.

    ``gains`` is ``[kp, kd, f_max]`` in a device array, not launch floats: a
    captured graph bakes a float but reads an array, so the studio's sliders
    can tune it live.
    """
    slot = wp.tid()
    if enabled[slot] == 0:
        return
    kp_pos = gains[0]
    kd_pos = gains[1]
    f_max = gains[2]
    seg_idx = seg_idx_arr[slot]
    gtf = body_q[grip_idx_arr[slot]]
    target = wp.transform_multiply(gtf, rel_tf_arr[slot])
    tpos = wp.transform_get_translation(target)
    stf = body_q[seg_idx]
    spos = wp.transform_get_translation(stf)
    vg = wp.spatial_top(body_qd[grip_idx_arr[slot]])
    wg = wp.spatial_bottom(body_qd[grip_idx_arr[slot]])
    gpos = wp.transform_get_translation(gtf)
    v_targ = vg + wp.cross(wg, spos - gpos)
    v_rel = wp.spatial_top(body_qd[seg_idx]) - v_targ
    force = kp_pos * (tpos - spos) - kd_pos * v_rel
    magnitude = wp.length(force)
    if magnitude > f_max:
        force = force * (f_max / magnitude)
    wp.atomic_add(body_f, seg_idx, wp.spatial_vector(force, wp.vec3(0.0, 0.0, 0.0)))


class _Hooks:
    """What ``build_hooks`` returns: the stages by phase, and the replayable latch state.

    ``slots`` is per world (one per hand); the arrays are ``num_worlds * slots``
    long, world-major, so slot ``s`` of world ``w`` is row ``w * slots + s``
    (:meth:`row`). A wrong row latches another world's rod. Callers write the
    replicated model's global body index for each world, not the template's.
    """

    def __init__(self, model, slots: int = 2, num_worlds: int = 1) -> None:
        device = model.device
        self.slots = int(slots)
        self.num_worlds = max(1, int(num_worlds))
        rows = self.slots * self.num_worlds
        self._hold_seg = wp.zeros(rows, dtype=wp.int32, device=device)
        self._hold_grip = wp.zeros(rows, dtype=wp.int32, device=device)
        self._hold_flag = wp.zeros(rows, dtype=wp.int32, device=device)
        self._hold_rel = wp.array([wp.transform_identity()] * rows, dtype=wp.transform, device=device)
        # One row for the whole fleet: the gains are a property of the coupling,
        # not of a world, and every world of a wave is running the same task.
        self._hold_gains = wp.zeros(3, dtype=wp.float32, device=device)

        def hold(state, dt) -> None:
            del dt
            wp.launch(
                hold_segment,
                dim=rows,
                inputs=[
                    state.body_q,
                    state.body_qd,
                    state.body_f,
                    self._hold_seg,
                    self._hold_grip,
                    self._hold_rel,
                    self._hold_flag,
                    self._hold_gains,
                ],
                device=device,
            )

        self.stages = {"rigid_force": [hold]}
        self.aux_sinks = {
            "hold_flag": self._hold_flag,
            "hold_seg": self._hold_seg,
            "hold_grip": self._hold_grip,
            "hold_rel": self._hold_rel,
            "hold_gains": self._hold_gains,
        }

    def row(self, slot: int, world: int = 0) -> int:
        """Row of ``slot`` in ``world`` (world-major; the identity for one world)."""
        return int(world) * self.slots + int(slot)


def build_hooks(model, params: dict, *, slots: int = 2, num_worlds: int = 1) -> _Hooks:
    """The latch and its sinks. ``params["hold_gains"]`` seeds ``[kp, kd, f_max]``.

    Args:
        slots: Latch slots per world -- one per hand.
        num_worlds: Worlds in the model. Above one the arrays are sized for the
            fleet and the launch covers it; the kernel is unchanged.
    """
    hooks = _Hooks(model, slots=slots, num_worlds=num_worlds)
    gains = params.get("hold_gains")
    if gains is not None:
        hooks._hold_gains.assign([float(v) for v in gains])
    return hooks
