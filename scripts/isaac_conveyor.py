"""Test contact transport on an existing USD belt and an existing rigid body."""

import argparse
import csv
import json
from pathlib import Path
import sys

from isaac_common import runtime_version, save_report, start_app, wait_for_stage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--belt-prim", required=True, help="Collision + kinematic rigid body prim with surface velocity")
    parser.add_argument("--body-prim", required=True, help="Existing dynamic rigid body placed on the belt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, required=True, help="Positive test speed in m/s")
    parser.add_argument("--direction", type=float, nargs=3, default=[1, 0, 0], help="Direction in the belt's local frame")
    parser.add_argument("--phase-seconds", type=float, default=2.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    report_path = output / "conveyor_report.json"
    report = {"status": "running", "passed": False, "real_world_calibrated": False}
    save_report(report_path, report)
    app = None
    try:
        import numpy as np
        if not np.isfinite([args.speed, args.phase_seconds]).all() or min(args.speed, args.phase_seconds) <= 0:
            raise ValueError("--speed and --phase-seconds must be finite and positive")
        if args.phase_seconds < 0.1:
            raise ValueError("--phase-seconds must be at least 0.1 seconds")
        direction = np.asarray(args.direction, dtype=float)
        if not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
            raise ValueError("--direction must be finite and nonzero")
        direction /= np.linalg.norm(direction)
        app = start_app(args.headless)
        report["isaac_version"] = runtime_version()
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
        context = omni.usd.get_context()
        if not context.open_stage(str(args.input.resolve())):
            raise RuntimeError("USD stage could not be opened")
        stage = wait_for_stage(app, context)
        if not stage.HasAuthoredMetadata("metersPerUnit") or UsdGeom.GetStageUpAxis(stage) != "Z":
            raise ValueError("This runtime entrypoint requires explicitly authored units and a Z-up stage")
        units = UsdGeom.GetStageMetersPerUnit(stage)
        if not np.isfinite(units) or units <= 0:
            raise ValueError("Stage units must be finite and positive")
        belt, body = stage.GetPrimAtPath(args.belt_prim), stage.GetPrimAtPath(args.body_prim)
        if not belt or not body:
            raise ValueError("Belt or body prim does not exist")
        if not belt.HasAPI(UsdPhysics.CollisionAPI) or not belt.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Belt needs CollisionAPI and RigidBodyAPI on the specified prim")
        if not UsdPhysics.RigidBodyAPI(belt).GetKinematicEnabledAttr().Get():
            raise ValueError("Belt must be kinematic")
        if not body.HasAPI(UsdPhysics.RigidBodyAPI) or UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get():
            raise ValueError("Transported body must be dynamic")
        if not belt.GetAttribute("physxSurfaceVelocity:surfaceVelocityEnabled").Get():
            raise ValueError("Configure the belt with real2sim prepare-usd before running")
        matrix = UsdGeom.Xformable(belt).ComputeLocalToWorldTransform(0)
        world_direction = np.array(matrix.TransformDir(Gf.Vec3d(*direction)))
        world_direction /= np.linalg.norm(world_direction)
        surface = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(belt)
        surface.CreateSurfaceVelocityLocalSpaceAttr(True)
        velocity = surface.CreateSurfaceVelocityAttr()
        velocity.Set(Gf.Vec3f(0))
        world = World(stage_units_in_meters=units, physics_dt=1 / 120, rendering_dt=1 / 60)
        tracked = world.scene.add(SingleRigidPrim(prim_path=args.body_prim, name="transported_body"))
        world.reset()
        metrics = {}
        elapsed = 0.0
        with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time_s", "phase", "speed_m_s", "x_m", "y_m", "z_m"])
            for phase, speed, duration in [("settle", 0, 1), ("forward", args.speed, args.phase_seconds),
                                            ("reverse", -args.speed, args.phase_seconds), ("stop", 0, 1)]:
                velocity.Set(Gf.Vec3f(*(direction * speed / units)))
                start = np.asarray(tracked.get_world_pose()[0], dtype=float) * units
                samples = []
                for _ in range(round(duration * 120)):
                    if not app.is_running():
                        raise RuntimeError("Simulation window closed before verification completed")
                    world.step(render=not args.headless)
                    elapsed += 1 / 120
                    position = np.asarray(tracked.get_world_pose()[0], dtype=float) * units
                    if not np.isfinite(position).all():
                        raise RuntimeError("Transported body has a non-finite physics pose")
                    writer.writerow([elapsed, phase, speed, *position])
                    samples.append(position)
                delta = position - start
                metrics[phase] = {"distance_along_belt_m": float(delta @ world_direction),
                                  "final_position_m": position.tolist(),
                                  "height_span_m": float(np.ptp(np.array(samples)[:, 2]))}
        minimum = max(0.01, args.speed * args.phase_seconds * 0.15)
        checks = {"forward_transport": metrics["forward"]["distance_along_belt_m"] > minimum,
                  "reverse_transport": metrics["reverse"]["distance_along_belt_m"] < -minimum,
                  "remained_at_belt_height": all(metrics[p]["height_span_m"] < 0.05 for p in ("forward", "reverse", "stop")),
                  "stopped": abs(metrics["stop"]["distance_along_belt_m"]) < 0.02}
        report.update(status="passed" if all(checks.values()) else "failed", passed=all(checks.values()),
                      checks=checks, metrics=metrics, speed_m_s=args.speed, trajectory="trajectory.csv",
                      scope="Contact transport of the supplied body on the supplied belt; no robot or throughput validation")
        save_report(report_path, report)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 2
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=str(error), passed=False)
        save_report(report_path, report)
        print(str(error), file=sys.stderr)
        return 2
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
