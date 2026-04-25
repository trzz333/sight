"""HSV-threshold perception for Signal Dodge.

Inputs are BGR uint8 frames (see capture/region.py for the normalization contract).
Outputs a dict with player center/bbox and a list of hazard centers/bboxes. Downstream code
(policy, logger) never touches the frame ndarray directly.
"""

from .vision import PerceptionResult, perceive_frame

__all__ = ["PerceptionResult", "perceive_frame"]
