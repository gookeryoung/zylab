"""core.project 工程文件（HDF5）测试."""

from __future__ import annotations

import numpy as np
import pytest

from zylab.core.errors import ProjectFileError
from zylab.core.project import PROJECT_SCHEMA_VERSION, Project


def test_create_open_close_roundtrip(tmp_path) -> None:
    """创建 → 打开 → 关闭应完整走通."""
    path = tmp_path / "test.zprj"
    with Project.create(path, name="roundtrip") as proj:
        assert proj.path == path
        meta = proj.meta
        assert meta["name"] == "roundtrip"
        assert meta["schema_version"] == PROJECT_SCHEMA_VERSION
    # 二次打开（只读）
    with Project.open(path) as proj2:
        assert proj2.meta["name"] == "roundtrip"


def test_array_roundtrip(tmp_path) -> None:
    """数组写入后应可原样读出."""
    path = tmp_path / "array.zprj"
    original = np.array([[1.0, 2.0], [3.0, 4.0]])
    with Project.create(path) as proj:
        proj.write_array("results", "displacement", original, attrs={"unit": "m"})

    with Project.open(path) as proj:
        loaded = proj.read_array("results", "displacement")
        np.testing.assert_array_equal(loaded, original)
        # 属性需通过底层访问（暂不在公共 API 暴露）


def test_json_roundtrip(tmp_path) -> None:
    """JSON 对象写入后应可原样读出."""
    path = tmp_path / "json.zprj"
    data = {"materials": ["steel", "aluminum"], "settings": {"tolerance": 1e-6}}
    with Project.create(path) as proj:
        proj.write_json("settings", "solver", data)

    with Project.open(path) as proj:
        loaded = proj.read_json("settings", "solver")
        assert loaded == data


def test_list_names(tmp_path) -> None:
    """list_names 应按名称排序返回成员."""
    path = tmp_path / "list.zprj"
    with Project.create(path) as proj:
        proj.write_array("results", "c_field", np.zeros(1))
        proj.write_array("results", "a_field", np.ones(1))
        proj.write_array("results", "b_field", np.full(1, 2.0))
    with Project.open(path) as proj:
        assert proj.list_names("results") == ["a_field", "b_field", "c_field"]


def test_read_missing_dataset(tmp_path) -> None:
    """读取不存在的数据集应抛 ProjectFileError."""
    path = tmp_path / "missing.zprj"
    with Project.create(path) as proj, pytest.raises(ProjectFileError, match="数据集不存在"):
        proj.read_array("results", "nope")


def test_read_missing_group(tmp_path) -> None:
    """写入不存在的组应自动创建；读取不存在的组成员应抛 ProjectFileError."""
    path = tmp_path / "group.zprj"
    with Project.create(path) as proj:
        proj.write_array("new_group", "data", np.zeros(1))
    with Project.open(path) as proj, pytest.raises(ProjectFileError, match="数据集不存在"):
        proj.read_array("new_group", "absent")


def test_file_not_exists(tmp_path) -> None:
    """打开不存在的文件应抛 ProjectFileError."""
    path = tmp_path / "not_exists.zprj"
    with pytest.raises(ProjectFileError, match="工程文件不存在"):
        Project.open(path)


def test_invalid_file(tmp_path) -> None:
    """打开非 HDF5 文件应抛 ProjectFileError."""
    path = tmp_path / "not_hdf5.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ProjectFileError, match="工程文件无法打开"):
        Project.open(path)


def test_schema_version_mismatch(tmp_path) -> None:
    """主版本号不匹配应拒绝打开."""
    import h5py

    path = tmp_path / "bad_schema.zprj"
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("meta")
        meta.attrs["schema_version"] = "99.0.0"
    with pytest.raises(ProjectFileError, match="schema 版本不兼容"):
        Project.open(path)


def test_array_overwrite(tmp_path) -> None:
    """同名数组应被覆盖而非追加."""
    path = tmp_path / "overwrite.zprj"
    with Project.create(path) as proj:
        proj.write_array("results", "val", np.array([1, 2, 3]))
        proj.write_array("results", "val", np.array([4, 5]))
    with Project.open(path) as proj:
        arr = proj.read_array("results", "val")
        assert list(arr) == [4, 5]


def test_write_array_invalid_data(tmp_path) -> None:
    """object dtype 数组无法写入 HDF5，应抛 ProjectFileError."""
    path = tmp_path / "bad_array.zprj"
    with Project.create(path) as proj, pytest.raises(ProjectFileError, match="数组写入失败"):
        proj.write_array("results", "bad", object())


def test_read_json_invalid(tmp_path) -> None:
    """JSON 解析失败应抛 ProjectFileError（故障注入：直接写非 JSON 文本，访问私有 _h5）."""
    path = tmp_path / "bad_json.zprj"
    with Project.create(path) as proj:
        proj._h5["settings"].create_dataset("broken", data="{not json")
        with pytest.raises(ProjectFileError, match="JSON 数据集解析失败"):
            proj.read_json("settings", "broken")


def test_require_group_conflict(tmp_path) -> None:
    """组名与已有数据集冲突应抛 ProjectFileError."""
    path = tmp_path / "conflict.zprj"
    with Project.create(path) as proj:
        # 在 root 下直接创建同名 dataset，与后续 require_group("mat") 冲突
        proj._h5.create_dataset("mat", data=b"x")
        with pytest.raises(ProjectFileError, match="组名与已有数据集冲突"):
            proj._require_group("mat")


def test_read_array_on_group(tmp_path) -> None:
    """对 Group 节点读数组应抛 ProjectFileError（故障注入：直接建子组，访问私有 _h5）."""
    path = tmp_path / "group_node.zprj"
    with Project.create(path) as proj:
        proj._h5["model"].create_group("subgroup")
        with pytest.raises(ProjectFileError, match="目标不是数据集"):
            proj.read_array("model", "subgroup")


def test_close_idempotent(tmp_path) -> None:
    """close 幂等，重复调用不抛异常."""
    proj = Project.create(tmp_path / "close.zprj")
    proj.close()
    proj.close()


def test_create_oserror(tmp_path, monkeypatch) -> None:
    """底层 HDF5 创建失败应包装为 ProjectFileError."""
    import zylab.core.project as project_mod

    def boom(*args, **kwargs):
        raise OSError("磁盘故障")

    monkeypatch.setattr(project_mod.h5py, "File", boom)
    with pytest.raises(ProjectFileError, match="工程文件创建失败"):
        Project.create(tmp_path / "x.zprj")
