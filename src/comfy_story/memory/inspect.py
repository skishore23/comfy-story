"""Inspect pinned memory weights without loading ComfyUI or a foundation model."""

from __future__ import annotations

import argparse
import json

from comfy_story.memory.health import history_effect_report
from comfy_story.memory.settings import InspectionCodec, MemoryConfiguration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Authenticate an H3 memory checkpoint and test its history readout"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--foundation-sha256", required=True)
    parser.add_argument("--model-configuration-sha256", required=True)
    parser.add_argument("--vae-sha256", required=True)
    args = parser.parse_args(argv)
    configuration = MemoryConfiguration(
        args.checkpoint,
        args.sha256,
        args.foundation_sha256,
        args.model_configuration_sha256,
        args.vae_sha256,
    )
    try:
        runtime = configuration.load(InspectionCodec())
        try:
            report = {
                "configuration": configuration.binding(),
                "checkpoint_authenticated": True,
                "history_effect": history_effect_report(runtime.bridge, require_effect=True),
                "foundation_and_vae_loaded": False,
                "rendered_quality_validated": False,
            }
        finally:
            runtime.close()
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
