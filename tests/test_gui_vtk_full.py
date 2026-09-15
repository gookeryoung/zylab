"""最终冲刺：覆盖 vtk_view.py 剩余 13 行 — 简化版."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_vtk_view_full_flow(qapp, monkeypatch, tmp_path: Path) -> None:
    """覆盖 load_polydata(90) + load_mesh_file(71-74) + load_unstructured(83-86) + _attach_to_layout."""
    import sys

    # --- Mock pyvistaqt.QtInteractor 覆盖 _attach_to_layout (55-61) ---
    from PySide2.QtWidgets import QWidget

    from zylab.gui.widgets.vtk_view import VtkView

    class FakeRenderer:
        def clear(self):
            pass

        def add_mesh(self, m):
            pass

        def reset_camera(self):
            pass

    class FakeInteractor(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.renderer = FakeRenderer()

        def render(self):
            pass

    fake_pvqt = SimpleNamespace(QtInteractor=FakeInteractor)
    monkeypatch.setitem(sys.modules, "pyvistaqt", fake_pvqt)

    # --- Mock pyvista 覆盖 load_mesh_file + load_unstructured ---
    class FakePV:
        @staticmethod
        def read(_p):
            return object()

        class UnstructuredGrid:
            def __init__(self, c, p):
                pass

    monkeypatch.setitem(sys.modules, "pyvista", FakePV)

    # --- 创建 VtkView ---
    v = VtkView()

    # load_polydata (line 90)
    v.load_polydata(object())

    # load_mesh_file (line 71, 73, 74)
    mesh_path = tmp_path / "dummy.vtk"
    mesh_path.write_text("dummy", encoding="utf-8")
    v.load_mesh_file(str(mesh_path))

    # load_unstructured_grid (line 83, 85, 86)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    cells = {10: np.array([0, 1, 2, 3])}
    v.load_unstructured_grid(points, cells)

    # 确保 _attach_to_layout 也跑了（通过 _show_mesh → _ensure_interactor）
    assert v._interactor is not None
