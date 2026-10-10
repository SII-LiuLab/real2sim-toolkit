"""Import prepared MJCF with Isaac Sim 5.x and package the resulting USD."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

from isaac_common import add_toolkit_path, runtime_version, save_report, start_app, wait_for_stage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Folder produced by real2sim isaac-prepare")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--floating", action="store_true", help="Import a free base instead of a fixed cabinet/equipment base")
    parser.add_argument("--self-collision", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    report_path = output / "import_report.json"
    report = {"status": "running", "isaac_runtime_validated": False}
    save_report(report_path, report)
    app = None
    try:
        source = args.input.resolve()
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "prepared" or manifest.get("inertia_preserved_in_mujoco") is not True:
            raise ValueError("Input must be a successful isaac-prepare export")
        staging = output / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        staged_source = staging / "mjcf_source"
        if source == staged_source or output == source or source in output.parents:
            raise ValueError("Use an output folder outside the prepared MJCF input")
        shutil.copytree(source, staged_source, dirs_exist_ok=True)
        app = start_app(args.headless)
        report["isaac_version"] = runtime_version()
        import omni.kit.app
        import omni.kit.commands
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)
        app.update()
        # The importer needs a saved stage identifier for texture paths.
        stage_file = staging / "import.usda"
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        if not stage.GetRootLayer().Export(str(stage_file)):
            raise RuntimeError("Could not save the import stage")
        del stage
        context = omni.usd.get_context()
        if not context.open_stage(str(stage_file)):
            raise RuntimeError("Could not open the saved import stage")
        stage = wait_for_stage(app, context)
        ok, config = omni.kit.commands.execute("MJCFCreateImportConfig")
        if not ok:
            raise RuntimeError("MJCF import configuration command failed")
        config.set_fix_base(not args.floating)
        config.set_make_default_prim(False)
        config.set_import_inertia_tensor(True)
        config.set_distance_scale(1.0)
        config.set_self_collision(args.self_collision)
        config.set_create_physics_scene(False)
        ok, _ = omni.kit.commands.execute("MJCFCreateAsset", mjcf_path=str(staged_source / "model.xml"),
                                          import_config=config, prim_path="/World/Asset")
        if not ok:
            raise RuntimeError("MJCF importer failed")
        stage = wait_for_stage(app, context)
        if not stage.GetPrimAtPath("/World/Asset"):
            raise RuntimeError("Importer did not produce /World/Asset")
        joint_map = {"PhysicsRevoluteJoint": "hinge", "PhysicsPrismaticJoint": "slide", "PhysicsSphericalJoint": "ball"}
        joints = [{"path": str(p.GetPath()), "type": joint_map[p.GetTypeName()]} for p in stage.Traverse() if p.GetTypeName() in joint_map]
        expected = Counter(j["type"] for j in manifest["joints"] if j["type"] != "free")
        if Counter(j["type"] for j in joints) != expected:
            raise RuntimeError(f"Imported movable joint topology differs from MJCF: {joints}")
        scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(9.81)
        stage.GetRootLayer().Save()
        add_toolkit_path()
        from robocasa_real2sim.usd_backend import inspect_usd, prepare_usd
        prepare_usd(stage_file, output / "asset")
        structure = inspect_usd(output / "asset/asset.usda")
        report.update(status="imported", asset="asset/asset.usda", imported_joints=joints,
                      structure=structure, dynamics_tested=False,
                      note="Import and topology checked; joint motion, contact, materials and inertia need runtime review")
        save_report(report_path, report)
        print(json.dumps(report, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=str(error), isaac_runtime_validated=False)
        save_report(report_path, report)
        print(str(error), file=sys.stderr)
        return 2
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    raise SystemExit(main())
