# Isaac execution

Run CPU tooling in the toolkit environment. Run simulator scripts from the checkout with Isaac's `python.sh`. The adapters target the Isaac Sim 5.x PhysX profile using documented 5.1 APIs. Isaac 6.x importer/backend changes require adaptation; USD parsing alone does not prove compatibility.

## MJCF

Run `real2sim isaac-prepare INPUT_XML --output PREPARED_FOLDER`. It bundles files, names anonymous bodies/joints, and freezes MuJoCo-computed mass/COM/inertia. It requires expanded MJCF and a separate output folder. The manifest records topology and inertia preservation in MuJoCo.

Then run `ISAAC_ROOT/python.sh scripts/isaac_import.py --input PREPARED_FOLDER --output IMPORT_FOLDER`, adding `--headless` for a server. The script saves the stage before import so textures can resolve, imports in metres, checks movable joint type counts, and packages USD. Use `--floating` for a free base. Self collision defaults off; enabling it requires contact inspection.

The output is `IMPORT_FOLDER/asset/asset.usda`. `import_report.json` distinguishes import from dynamics. Review axes, limits, clearance, inertia and materials. Drive joints and measure stability/contact in Isaac before reporting successful articulated behavior; general joint-motion validation is not implemented by the importer.

## Existing USD and belts

`real2sim prepare-usd INPUT_USD --output ASSET_FOLDER` flattens local composition and bundles file attributes. Units/up axis must be explicit. Animated file paths need dedicated packaging. Resolve remote/missing dependencies locally first.

To configure a belt, also supply `--belt-prim USD_PATH --speed M_PER_S --direction X Y Z`. The collision surface becomes a kinematic rigid body with `PhysxSurfaceVelocityAPI`. Avoid nested/conflicting rigid bodies and joint constraints; a separate valid surface is needed when the frame is articulated. Velocity converts from m/s to stage units. This models contact transport, not belt deformation or detailed motor/roller dynamics.

Your stage must already contain a dynamic object placed on the belt. Run `ISAAC_ROOT/python.sh scripts/isaac_conveyor.py --input ASSET_USD --belt-prim BELT_PATH --body-prim BODY_PATH --speed M_PER_S --output RUN_FOLDER`, with optional `--phase-seconds` and `--headless`. Prim paths refer to the composed stage. The runner requires Z up and explicit units.

It settles the object, commands forward/reverse/stop velocities, reads physics poses through `SingleRigidPrim`, and writes `trajectory.csv` plus `conveyor_report.json`. Choose duration/speed that keeps the body on the belt. Checks cover direction, stopping and height; inspect slip, collisions, edge drops and task-specific tolerances. Passing does not calibrate speed/friction or validate robot interaction.

Keep `prepared`, `imported` and physically `passed` distinct. SDK/version/import/transport failures return failure reports and nonzero exit status. CPU checks cannot replace runtime tests.

Primary sources: [MJCF importer](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_mjcf.html), [rigid-body API](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.prims/docs/index.html), [conveyor utility](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/digital_twin/warehouse_logistics/ext_isaacsim_asset_gen_conveyor.html), [6.0 changes](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/importer_exporter/ext_isaacsim_asset_importer_mjcf.html).
