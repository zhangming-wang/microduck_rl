"""Restore MJCF colors on an Isaac Lab USD imported with instanceable meshes.

Run with Isaac Sim's Python/USD libraries available (``pxr``). Creates a sibling
copy; the imported USD and its generated layers are left untouched.
"""

import argparse
import math
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from pxr import Usd, UsdPhysics, UsdShade


def _expected_colors(mjcf: Path) -> tuple[dict[str, tuple[float, ...]], Counter]:
    root = ET.parse(mjcf).getroot()
    colors = {
        material.get("name"): tuple(float(v) for v in material.get("rgba").split()[:3])
        for material in root.findall("./asset/material")
    }
    expected = {}
    counts = Counter()
    for geom in root.findall(".//worldbody//geom"):
        if geom.get("class") != "visual" or not geom.get("mesh"):
            continue
        mesh, material = geom.get("mesh"), geom.get("material")
        if material not in colors:
            raise ValueError(f"No RGB material for {mesh}: {material}")
        if mesh in expected and expected[mesh] != colors[material]:
            raise ValueError(f"Mesh {mesh} uses different MJCF colors")
        expected[mesh] = colors[material]
        counts[mesh] += 1
    if not counts:
        raise ValueError(f"No visual mesh geoms found in {mjcf}")
    return expected, counts


def _visual_meshes(stage: Usd.Stage):
    root = str(stage.GetDefaultPrim().GetPath())
    return [
        prim
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if prim.GetTypeName() == "Mesh"
        and str(prim.GetPath()).startswith(root + "/")
        and "/visuals/" in str(prim.GetPath())
    ]


def _bound_rgb(prim) -> tuple[float, ...]:
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        raise ValueError(f"No material bound to {prim.GetPath()}")
    colors = [
        child.GetAttribute("inputs:diffuse_color_constant").Get()
        for child in material.GetPrim().GetChildren()
        if child.IsA(UsdShade.Shader) and child.GetAttribute("inputs:diffuse_color_constant")
    ]
    if len(colors) != 1:
        raise ValueError(f"Expected one diffuse color on {material.GetPath()}")
    return tuple(colors[0])


def _check_colors(stage: Usd.Stage, expected: dict, counts: Counter, parent: bool) -> None:
    meshes = _visual_meshes(stage)
    if Counter(prim.GetName() for prim in meshes) != counts:
        raise ValueError("Visual USD meshes do not match the MJCF mesh names and counts")
    for prim in meshes:
        target = prim.GetParent() if parent else prim
        actual = _bound_rgb(target)
        if not all(math.isclose(a, b, abs_tol=1e-5) for a, b in zip(actual, expected[prim.GetName()])):
            raise ValueError(f"Wrong color for {prim.GetPath()}: {actual} (expected {expected[prim.GetName()]})")


def _physics_prims(stage: Usd.Stage) -> tuple[list, list]:
    prims = Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
    joints, colliders = [], []
    for prim in prims:
        if prim.IsA(UsdPhysics.Joint):
            joints.append((str(prim.GetPath()), prim.GetTypeName()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append((str(prim.GetPath()), prim.GetTypeName()))
    return sorted(joints), sorted(colliders)


def fix(mjcf: Path, usd: Path, output_dir: Path) -> Path:
    mjcf, usd, output_dir = mjcf.resolve(), usd.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    expected, counts = _expected_colors(mjcf)
    original = Usd.Stage.Open(str(usd))
    if original is None:
        raise ValueError(f"Cannot open {usd}")
    _check_colors(original, expected, counts, parent=True)
    original_physics = _physics_prims(original)

    # The importer binds the right material to each visual geom, but a white/gray
    # binding on its shared mesh prototype overrides it. Remove only that fallback.
    shutil.copytree(usd.parent, output_dir, ignore=shutil.ignore_patterns(".asset_hash", "config.yaml"))
    base_usd = output_dir / "configuration" / f"{usd.stem}_base.usd"
    stage = Usd.Stage.Open(str(base_usd))
    if stage is None:
        raise ValueError(f"Cannot open {base_usd}")
    stage.SetEditTarget(stage.GetRootLayer())
    prototype_root = stage.GetPrimAtPath("/meshes")
    prototypes = [
        prim for prim in Usd.PrimRange(prototype_root) if prim.GetTypeName() == "Mesh"
    ]
    if {prim.GetName() for prim in prototypes} != set(expected):
        raise ValueError("USD mesh prototypes do not match MJCF visual mesh names")
    for prim in prototypes:
        binding = prim.GetRelationship("material:binding")
        targets = binding.GetTargets() if binding else []
        if len(targets) != 1 or not targets[0].name.startswith("DefaultMaterial"):
            raise ValueError(f"Unexpected prototype binding on {prim.GetPath()}: {targets}")
        prim.RemoveProperty("material:binding")
    stage.GetRootLayer().Save()

    corrected = output_dir / usd.name
    result = Usd.Stage.Open(str(corrected))
    if result is None:
        raise ValueError(f"Cannot open corrected USD: {corrected}")
    _check_colors(result, expected, counts, parent=False)
    if _physics_prims(result) != original_physics:
        raise ValueError("Joint or collision prims changed while restoring colors")
    print(f"Restored {len(counts)} mesh colors across {sum(counts.values())} visual geoms: {corrected}")
    return corrected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mjcf", type=Path)
    parser.add_argument("usd", type=Path, help="Top-level USD produced by the MJCF importer")
    parser.add_argument("--output-dir", type=Path, help="New sibling output directory (never overwritten)")
    args = parser.parse_args()
    fix(args.mjcf, args.usd, args.output_dir or args.usd.parent.with_name(args.usd.parent.name + "_colored"))
