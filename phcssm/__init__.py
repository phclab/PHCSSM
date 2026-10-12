# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright National Yang Ming Chiao Tung University, Taiwan.
# Creator Po-Han Chiang. Patent pending.
# Noncommercial use only. For commercial licensing see LICENSE.
"""PHCSSM: a parallelized hierarchical connectome for spiking state-space models."""

from . import config_io, model, trainer

__all__ = ["config_io", "model", "trainer"]
