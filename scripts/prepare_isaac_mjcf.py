"""Create a sibling MJCF that Isaac Sim can import (one top-level default)."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def prepare(source: Path) -> Path:
    source = source.resolve()
    output = source.with_name(f"{source.stem}_isaac.xml")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    # Retain comments and class order; MJCF accepts separate top-level defaults,
    # but Isaac Sim's MJCF importer currently accepts only one.
    tree = ET.parse(source, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError(f"Expected a MuJoCo model, got <{root.tag}>")
    defaults = root.findall("default")
    if not defaults:
        raise ValueError("MJCF contains no top-level <default>")
    for block in defaults[1:]:
        for child in block:
            defaults[0].append(child)
        root.remove(block)

    root.insert(0, ET.Comment(f" Isaac Sim import copy of {source.name}; merge top-level defaults only. "))
    ET.indent(tree, space="  ")
    with output.open("xb") as stream:
        tree.write(stream, encoding="utf-8", xml_declaration=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source MJCF; output goes beside it so meshdir stays valid")
    args = parser.parse_args()
    print(prepare(args.source))
