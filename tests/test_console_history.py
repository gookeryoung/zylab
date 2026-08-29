"""console.history 命令历史测试."""

from __future__ import annotations

from zylab.console import CommandHistory


def test_add_and_entries() -> None:
    """add 应按顺序追加."""
    history = CommandHistory()
    history.add("a = 1")
    history.add("b = 2")
    assert history.entries == ["a = 1", "b = 2"]


def test_add_skips_empty_and_continuous_duplicate() -> None:
    """空命令与连续重复不入库."""
    history = CommandHistory()
    history.add("x = 1")
    history.add("x = 1")
    history.add("")
    history.add("   ")
    assert history.entries == ["x = 1"]


def test_add_truncates_to_maxsize() -> None:
    """超出 maxsize 时丢弃最旧条目."""
    history = CommandHistory(maxsize=3)
    for i in range(5):
        history.add(f"cmd{i}")
    assert history.entries == ["cmd2", "cmd3", "cmd4"]


def test_previous_next_navigation() -> None:
    """上翻暂存当前输入，下翻到底还原暂存."""
    history = CommandHistory()
    history.add("first")
    history.add("second")
    assert history.previous("当前输入") == "second"
    assert history.previous() == "first"
    assert history.previous() is None  # 已到最旧
    assert history.next() == "second"
    assert history.next() == "当前输入"  # 还原暂存
    assert history.next() is None  # 游标已复位


def test_navigation_on_empty_history() -> None:
    """空历史导航返回 None."""
    history = CommandHistory()
    assert history.previous() is None
    assert history.next() is None


def test_add_resets_cursor() -> None:
    """浏览中新输入入库后游标复位."""
    history = CommandHistory()
    history.add("a")
    history.previous()
    history.add("b")
    assert history.previous() == "b"


def test_load_save_roundtrip(tmp_path) -> None:
    """保存后可完整加载."""
    path = tmp_path / "history.json"
    history = CommandHistory(path)
    history.add("x = 1")
    history.add("plot(x)")
    history.save()
    loaded = CommandHistory(path)
    loaded.load()
    assert loaded.entries == ["x = 1", "plot(x)"]


def test_load_missing_file(tmp_path) -> None:
    """文件缺失时静默忽略."""
    history = CommandHistory(tmp_path / "none.json")
    history.load()  # 不抛异常
    assert history.entries == []


def test_load_corrupted_file(tmp_path) -> None:
    """文件损坏时静默忽略并保留空历史."""
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    history = CommandHistory(path)
    history.load()
    assert history.entries == []


def test_load_non_list_json(tmp_path) -> None:
    """JSON 非数组时忽略内容."""
    path = tmp_path / "dict.json"
    path.write_text('{"k": 1}', encoding="utf-8")
    history = CommandHistory(path)
    history.load()
    assert history.entries == []


def test_save_without_path() -> None:
    """未配置路径时 save 静默跳过."""
    CommandHistory().save()  # 不抛异常


def test_save_creates_parent_dirs(tmp_path) -> None:
    """保存时自动创建父目录."""
    path = tmp_path / "deep" / "nested" / "history.json"
    history = CommandHistory(path)
    history.add("cmd")
    history.save()
    assert path.exists()
