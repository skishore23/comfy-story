"""Supported MiniMax H3 generation strategies."""

from enum import StrEnum


class StorySampler(StrEnum):
    """MiniMax generation strategy recorded independently from memory identity."""

    NATIVE_RES_MULTISTEP = "native-res-multistep"
    SPEED_EULER_2STAGE = "speed-euler-2stage"
    TURBO_4STEP = "turbo-4step"
    TURBO_8STEP = "turbo-8step"
    FULL_HD_2PASS = "full-hd-2pass"
    NVFP4_EXACT = "nvfp4-exact"
    NVFP4_BALANCED = "nvfp4-balanced"
    NVFP4_ULTRA_FAST = "nvfp4-ultra-fast"
    NVFP4_TURBO = "nvfp4-turbo-4step"
