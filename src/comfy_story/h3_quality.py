"""Versioned H3 spatial refinement preset, distinct from SPEED sampling."""

FULL_HD_SAMPLER = "Full HD 2-pass"
FULL_HD_PROTOCOL = "duet-h3-full-hd-v1"
UPSCALER_MODEL = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
UPSCALER_NODE_REVISION = "d7c01b9011f2e8439493f6c02c29995a27df276f"
UPSCALER_SOURCE_SHA256 = "744063b43e0f3eec23e2485cb7c65503069946ca9690906ecb548d7515cb89e2"
REFINEMENT_SIGMAS = (6 / 7, 2 / 3, 6 / 13, 0.0)
FULL_HD_CONFIGURATION = {
    "protocol": FULL_HD_PROTOCOL,
    "output_width": 1920,
    "output_height": 1088,
    "upscaler": UPSCALER_MODEL,
    "upscaler_node_revision": UPSCALER_NODE_REVISION,
    "upscaler_source_sha256": UPSCALER_SOURCE_SHA256,
    "upscale_precision": "fp16",
    "refinement_sigmas": REFINEMENT_SIGMAS,
    "refinement_seed_offset": 10000,
    "decode_tile": 256,
    "decode_overlap": 32,
}
