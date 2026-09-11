"""突破 95% 覆盖率 — 极简错误路径测试."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.fea import ElementBlock, ElementError, ElementType, LinearElastic, MeshError


class TestMaterialDensity:
    """fea.material 负密度校验 (line 50)."""

    def test_density_negative_raises(self) -> None:
        with pytest.raises(ElementError, match="质量密度须非负"):
            LinearElastic(200e9, 0.3, density=-1.0)


class TestElementBlockConn:
    """fea.mesh 连接表维度校验 (line 91)."""

    def test_conn_1d_raises(self) -> None:
        flat = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
        with pytest.raises(MeshError, match="须为二维数组"):
            ElementBlock(etype=ElementType.TRIA3, conn=flat, material=0, section=0)
