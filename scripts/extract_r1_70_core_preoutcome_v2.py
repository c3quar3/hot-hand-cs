"""Versioned loader repair for the R1 core preoutcome extractor."""

from __future__ import annotations

import sys

import extract_r1_70_core_preoutcome_v1 as base

ORIGINAL_LOAD_PILOT = base.load_pilot


def load_pilot_with_frozen_sibling_path():
    source_directory = str(base.PILOT_SOURCE.parent)
    inserted = source_directory not in sys.path
    if inserted:
        sys.path.insert(0, source_directory)
    try:
        return ORIGINAL_LOAD_PILOT()
    finally:
        if inserted:
            sys.path.remove(source_directory)


base.load_pilot = load_pilot_with_frozen_sibling_path


if __name__ == "__main__":
    base.main()
