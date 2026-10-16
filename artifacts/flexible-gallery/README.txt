Five flexible assets / Real2sim

real2sim-flexible-assets.zip contains the five generated example assets,
their native scene, parameters, source/license records and file hashes.
Unzip it, then run in a compatible Real2sim + Newton runtime:

  real2sim agent --session native-session --device cuda:0 --renderer rtx --panels all --resume

The meshes use SI metres and Z up. OBJ contains visual geometry; the native
scene preserves flexible mechanics and contact settings. These are generated
examples, without real-object material calibration.

Report and actual simulation video:
  https://sii-liulab.github.io/real2sim-toolkit/flexible-reconstruction/

The report has no asset download links; the asset package is shared separately.
