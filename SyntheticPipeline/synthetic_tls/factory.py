"""Build concrete tree generators from configuration dicts."""

from __future__ import annotations

import sys
from typing import Any, Dict

from synthetic_tls.generators.arbaro import ArbaroGenerator
from synthetic_tls.generators.base import BaseTreeGenerator
from synthetic_tls.generators.blender import BlenderGenerator


def create_generator(gen_config: Dict[str, Any]) -> BaseTreeGenerator:
    """
    Construct a ``BaseTreeGenerator`` from a pipeline ``generator`` config block.

    Supported types: ``blender``, ``arbaro``.
    """
    gen_type = gen_config.get("type", "blender")

    if gen_type == "blender":
        blender_path = gen_config.get("blender_path")
        if not blender_path:
            print("Error: Generator type is 'blender' but 'blender_path' is missing in config.")
            print("Ensure your config defines generator -> blender_path.")
            sys.exit(1)
        return BlenderGenerator(
            blender_path=blender_path,
            blend_file=gen_config.get("blend_file"),
        )

    if gen_type == "arbaro":
        return ArbaroGenerator(
            jar_path=gen_config.get(
                "arbaro_jar_path", "SyntheticPipeline/lib/arbaro/arbaro_cmd.jar"
            ),
            xml_template=gen_config.get(
                "xml_template", "SyntheticPipeline/configs/arbaro.xml"
            ),
            java_bin=gen_config.get("java_path", "java"),
        )

    print(f"Error: Unsupported generator type '{gen_type}'. Available types: blender, arbaro.")
    sys.exit(1)
