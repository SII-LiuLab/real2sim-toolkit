# Adapted upstream components

Real2Sim incorporates selected scene, Newton flexible-domain, recording and UI
code adapted from [RoboSimStudio](https://github.com/robosimstudio/robosimstudio)
at commit `73739abe212f963e8a9f5862b27bca370df091a7`.

Copyright 2026 Kaiyuan Chen. Preserve the [Apache-2.0 license](third_party/robosimstudio/LICENSE),
[upstream notice](third_party/robosimstudio/NOTICE) and SPDX headers on adapted
files. The compatibility import namespaces `src/robosimstudio`,
`src/robosimstudio_studio` and `src/robosimstudio_real2sim_agent` are internal
components of the single Real2Sim distribution, not an independent checkout.

[UPSTREAM.json](third_party/robosimstudio/UPSTREAM.json) records the original
803-file baseline and its hashes for provenance. Only the selected runtime
code is retained; that baseline is not an inventory of this repository.
Tests, examples, training/benchmark tools, alternate-engine implementations
and robot model collections are not included. Runtime domain implementations
were moved into `src/robosimstudio/_domain_packs` and generic soft bodies use
Newton VBD. The source marker `src/robosimstudio/_real2sim_bundle.json` records
this adaptation; runtime reports fingerprint the current code separately.

These licenses apply to the respective upstream components and do not assign
a new license to the surrounding Real2Sim Toolkit code. Any separately supplied
asset retains its own license and provenance.
