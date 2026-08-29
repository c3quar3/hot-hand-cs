"""Versioned absent-context-event repair for the R1 core preoutcome extractor."""

from __future__ import annotations

import pandas as pd

import extract_r1_70_core_preoutcome_v2 as v2


ORIGINAL_EVENT_FRAME = v2.base.event_frame


def event_frame_with_empty_context(parser, name: str):
    try:
        return ORIGINAL_EVENT_FRAME(parser, name)
    except RuntimeError as exc:
        if str(exc) == f"unexpected_event_result:{name}" and name in v2.base.CONTEXT_EVENTS:
            return pd.DataFrame({"tick": pd.Series(dtype="int64")})
        raise


v2.base.event_frame = event_frame_with_empty_context


if __name__ == "__main__":
    v2.base.main()
