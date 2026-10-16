"""Portable, installable lock for the exercised Studio dependency combination.

This Python data resource ships in wheels without package-data configuration.
It locks the simulation, URDF/IK and rendering interfaces used by the harness,
not every unrelated package in the Python environment. GPU drivers and
system libraries are deliberately outside this distribution-metadata lock.
Bundled RSS keeps upstream provenance while its current source layout and
content may change; its fingerprint is evidence rather than an immutable pin.
"""

STUDIO_RUNTIME_LOCK = {
    "schema_version": 3,
    "profile": "embedded-studio-newton-1.6-python-3.12",
    "python": {"implementation": "CPython", "major": 3, "minor": 12,
               "observed_patch": "3.12.3"},
    "observed_platform": {"system": "Linux", "machine": "x86_64"},
    "required": {
        "robosimstudio": {
            "version": "0.1.0",
            "distribution": "robocasa-real2sim",
            "distribution_version": "0.1.0",
            "source": "embedded",
            "local_path": "src/robosimstudio",
            "module": "robosimstudio",
            "module_packages": ["robosimstudio", "robosimstudio_studio", "robosimstudio_real2sim_agent"],
            "source_paths": ["src/robosimstudio", "src/robosimstudio_studio",
                             "src/robosimstudio_real2sim_agent", "third_party/robosimstudio"],
            "origin_marker": "_real2sim_bundle.json",
            "integration": "real2sim embedded runtime",
            "upstream_provenance": {
                "project": "RoboSimStudio",
                "url": "https://github.com/robosimstudio/robosimstudio",
                "revision": "73739abe212f963e8a9f5862b27bca370df091a7",
                "original_source_digest": "cb5e2e291d484392bda5923a00b75158450e6f18feff166f49511f16eed5d2f7",
            },
        },
        "newton": {"version": "1.6.1", "extras": ["sim"]},
        "warp-lang": {"version": "1.18.0"},
        "mujoco": {"version": "3.12.0"},
        "mujoco-warp": {"version": "3.12.0"},
        "numpy": {"version": "2.5.4"},
        "scipy": {"version": "1.18.1"},
        "Pillow": {"version": "12.3.0"},
        "PyYAML": {"version": "6.0.3"},
        "imageio": {"version": "2.38.1", "extras": ["ffmpeg"]},
        "imageio-ffmpeg": {"version": "0.6.0"},
        "viser": {"version": "1.1.1"},
        "coacd": {"version": "1.0.11"},
        "fast-simplification": {"version": "0.2.0"},
        "pyroki": {
            "version": "0.0.0",
            "git_url": "https://github.com/chungmin99/pyroki.git",
            "git_revision": "95afccc22658c461ab1042a048ae4e9c24bc2a47",
            "editable_must_be_clean": True,
        },
        "jaxls": {
            "version": "0.0.0",
            "git_url": "https://github.com/brentyi/jaxls.git",
            "git_revision": "50a58be88c5ef74532f09e3f55268b4f02c490e3",
            "editable_must_be_clean": True,
        },
        "jax": {"version": "0.11.2"},
        "jaxlib": {"version": "0.11.2"},
        "jaxlie": {"version": "1.5.0"},
        "jax-dataclasses": {"version": "1.6.3"},
        "jaxtyping": {"version": "0.3.11"},
        "yourdfpy": {"version": "0.0.60"},
        "trimesh": {"version": "5.1.1"},
        "pycollada": {"version": "0.9.3"},
        "lxml": {"version": "6.1.3"},
    },
    "rtx_optional": {
        "ovrtx": {"version": "0.4.0.346409", "index_url": "https://pypi.nvidia.com"},
        "ovstage": {"version": "0.1.1.355824"},
        "usd-core": {"version": "26.8"},
        "pyglet": {"version": "2.1.19"},
    },
}
