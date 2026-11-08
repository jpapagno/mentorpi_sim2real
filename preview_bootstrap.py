"""Use standalone Kit's OpenUSD ABI before any Isaac Lab imports.

The Lab venv's USD wheel is suitable for headless Newton but is not binary
compatible with the USD plugins in this standalone Kit build. Keep this
bootstrap exclusive to preview; do not change headless training dependencies.
"""

import os
import importlib.machinery
import importlib.util
from pathlib import Path
import runpy
import sys


def configure_kit_usd():
    root = Path(os.environ["ISAAC_PATH"])
    candidates = sorted((root / "extscache").glob("omni.usd.libs-*/pxr/Tf"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one standalone USD runtime under {root}; found {len(candidates)}")
    usd_root = candidates[0].parent.parent
    if any(name == "pxr" or name.startswith("pxr.") for name in sys.modules):
        raise RuntimeError("OpenUSD loaded before the preview bootstrap; launch with preview-navigation.bat")
    sys.path.insert(0, str(usd_root))
    # Kit distributes pxr as a namespace package. The venv's regular pxr
    # package otherwise wins resolution even when Kit is first on sys.path.
    spec = importlib.machinery.ModuleSpec("pxr", loader=None, is_package=True)
    spec.submodule_search_locations = [str(usd_root / "pxr")]
    sys.modules["pxr"] = importlib.util.module_from_spec(spec)
    dll_dir = usd_root / "bin"
    handles = []
    if os.name == "nt":
        handles.append(os.add_dll_directory(str(dll_dir)))
    os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PXR_USD_WINDOWS_DLL_PATH"] = str(dll_dir)
    return usd_root, handles


if __name__ == "__main__":
    usd_root, dll_handles = configure_kit_usd()
    if "--check-runtime" in sys.argv:
        from pxr import Tf
        if not Path(Tf.__file__).resolve().is_relative_to(usd_root.resolve()):
            raise RuntimeError(f"Preview selected the wrong USD runtime: {Tf.__file__}")
        print(f"Validated preview OpenUSD: {Tf.__file__}")
    else:
        if os.environ.get("MENTORPI_PREVIEW_TRACE") == "1":
            import faulthandler
            faulthandler.dump_traceback_later(45, repeat=True)
        runpy.run_path(str(Path(__file__).with_name("run_navigation.py")), run_name="__main__")
