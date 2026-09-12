"""最后 2 行 report.py 覆盖测试 — 突破 95%."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from zylab.flowchart import report
from zylab.flowchart.errors import TemplateError

__all__ = []


class TestReportCloudFieldAndCoords:
    """report._cloud_field 不受支持场量 + _cloud_coords 无变形原坐标."""

    def test_cloud_field_unsupported_raises(self) -> None:
        """_cloud_field 收到未知 field → TemplateError."""
        fake = MagicMock()
        with (
            patch.object(report, "_auto_field", return_value="unsupported_field"),
            pytest.raises(TemplateError, match="不受支持"),
        ):
            report._cloud_field(fake, "", 3)

    def test_cloud_coords_passthrough(self) -> None:
        """_cloud_coords 无 displacement 无 mode_shapes → 原 mesh.coords."""
        fake = MagicMock()
        fake.mesh.n_nodes = 3
        fake.mesh.coords = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        fake.displacements = None
        fake.mode_shapes = None
        values = np.array([1.0, 2.0, 3.0])
        result = report._cloud_coords(fake, fake.mesh, values, 1.0)
        assert result.shape == (3, 2)
