"""WorkspaceManager 与 kernel cwd/cd 集成测试."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from zylab.console import ReplKernel
from zylab.core import EventBus, get_workspace_history_limit, update_runtime_config
from zylab.sci import (
    CURRENT_WORKSPACE_FILE,
    TOPIC_WORKSPACE_CHANGED,
    WorkspaceInfo,
    WorkspaceManager,
)


@pytest.fixture(autouse=True)
def _reset_runtime_history_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """每个测试前重置运行时工作区历史条数为默认值 10 + 恢复 os.chdir 到 cwd."""
    original_cwd = Path.cwd()
    update_runtime_config(workspace_history_limit=10)
    yield
    # 还原 cwd 到测试前（WorkspaceManager.set_workspace 内部会 os.chdir，
    # 跨测试如果 tmp_path 被清了，再 chdir 进去就会崩溃）
    os.chdir(original_cwd)


class TestWorkspaceManager:
    """WorkspaceManager 基础行为."""

    def test_default_cwd_is_path_cwd(self) -> None:
        """默认 cwd = Path.cwd().resolve()，且 source='init' 时不广播事件."""
        bus = EventBus()
        bus.subscribe(TOPIC_WORKSPACE_CHANGED, lambda _: (_ for _ in ()).throw(AssertionError("不应在 init 时广播")))
        wm = WorkspaceManager(bus)
        assert wm.cwd == Path.cwd().resolve()

    def test_set_workspace_valid_dir_publishes_event(self, tmp_path: Path) -> None:
        """成功切换应广播 TOPIC_WORKSPACE_CHANGED，payload 为 WorkspaceInfo."""
        bus = EventBus()
        received: list[WorkspaceInfo] = []
        bus.subscribe(TOPIC_WORKSPACE_CHANGED, received.append)
        wm = WorkspaceManager(bus)
        prev = wm.cwd

        target = tmp_path / "sub"
        target.mkdir()
        info = wm.set_workspace(target)

        assert wm.cwd == target.resolve()
        assert info.path == target.resolve()
        assert info.prev_path == prev
        assert info.source == "set"
        assert len(received) == 1
        assert received[0].path == target.resolve()
        # os.chdir 也应同步
        assert Path.cwd() == target.resolve()

    def test_set_workspace_nonexistent_silent_reject(self, tmp_path: Path) -> None:
        """路径不存在时静默拒绝，cwd 不变，不广播事件."""
        bus = EventBus()
        received: list[WorkspaceInfo] = []
        bus.subscribe(TOPIC_WORKSPACE_CHANGED, received.append)
        wm = WorkspaceManager(bus)
        original = wm.cwd

        target = tmp_path / "does_not_exist"
        info = wm.set_workspace(target)

        assert wm.cwd == original
        assert info.source == "invalid"
        assert len(received) == 0

    def test_set_workspace_same_dir_no_event(self, tmp_path: Path) -> None:
        """重复设置同一目录不广播事件."""
        bus = EventBus()
        received: list[WorkspaceInfo] = []
        bus.subscribe(TOPIC_WORKSPACE_CHANGED, received.append)
        target = tmp_path / "target"
        target.mkdir()
        wm = WorkspaceManager(bus)
        wm.set_workspace(target)
        n_after_first = len(received)

        wm.set_workspace(target)
        assert len(received) == n_after_first
        # 返回的 WorkspaceInfo.prev_path 为 None 表示无变化
        info = wm.set_workspace(target)
        assert info.prev_path is None

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        """save 写入 data_dir/workspace.json；load 恢复路径并触发事件."""
        import json

        bus = EventBus()
        data_dir = tmp_path / "state"
        wm = WorkspaceManager(bus, data_dir=data_dir)
        target = tmp_path / "my-work"
        target.mkdir()
        wm.set_workspace(target)

        save_path = wm.save()
        assert save_path == data_dir / CURRENT_WORKSPACE_FILE
        # 从 JSON 重新解析，避免 Windows 反斜杠转义差异
        saved_path = Path(json.loads(save_path.read_text(encoding="utf-8"))["path"])
        assert saved_path == target.resolve()

        # 切回一个不同的目录，确保 wm2 的 load 会触发真正的切换
        os.chdir(tmp_path)

        # 重建一个新 WM，验证 load 恢复路径
        bus2 = EventBus()
        restored: list[WorkspaceInfo] = []
        bus2.subscribe(TOPIC_WORKSPACE_CHANGED, restored.append)
        wm2 = WorkspaceManager(bus2, data_dir=data_dir)
        result = wm2.load()

        assert result is not None
        assert wm2.cwd == target.resolve()
        assert len(restored) == 1

    def test_load_skips_missing_file(self, tmp_path: Path) -> None:
        """data_dir 下无 workspace.json 时 load 返回 None，cwd 保持 Path.cwd()."""
        bus = EventBus()
        wm = WorkspaceManager(bus, data_dir=tmp_path)
        assert wm.load() is None
        assert wm.cwd == Path.cwd().resolve()

    def test_load_skips_corrupt_json(self, tmp_path: Path) -> None:
        """workspace.json 非法 JSON 时静默跳过."""
        data_dir = tmp_path / "state"
        data_dir.mkdir()
        (data_dir / CURRENT_WORKSPACE_FILE).write_text("not json", encoding="utf-8")
        wm = WorkspaceManager(data_dir=data_dir)
        assert wm.load() is None

    def test_load_skips_nonexistent_saved_path(self, tmp_path: Path) -> None:
        """持久化文件里路径已不存在时静默跳过."""
        data_dir = tmp_path / "state"
        data_dir.mkdir()
        ghost = tmp_path / "ghost-dir"
        (data_dir / CURRENT_WORKSPACE_FILE).write_text(f'{{"path": "{ghost}"}}', encoding="utf-8")
        wm = WorkspaceManager(data_dir=data_dir)
        assert wm.load() is None

    @pytest.mark.parametrize(
        "payload",
        [
            '{"path": ""}',
            '{"path": "   "}',
            '{"path": 123}',
            '{"path": null}',
        ],
    )
    def test_load_skips_invalid_path_field(self, tmp_path: Path, payload: str) -> None:
        """workspace.json 的 path 字段为空或非字符串时静默跳过."""
        data_dir = tmp_path / "state"
        data_dir.mkdir()
        (data_dir / CURRENT_WORKSPACE_FILE).write_text(payload, encoding="utf-8")
        wm = WorkspaceManager(data_dir=data_dir)
        assert wm.load() is None

    def test_save_failure_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """持久化写入失败时 save 返回 None（仅记日志不抛）."""
        wm = WorkspaceManager(data_dir=tmp_path)

        def _fail_write(*_args: object, **_kwargs: object) -> None:
            raise OSError("磁盘写入失败")

        monkeypatch.setattr(Path, "write_text", _fail_write)
        assert wm.save() is None

    def test_history_records_prev_cwd_capped_at_10(self, tmp_path: Path) -> None:
        """切换后旧 cwd 并入历史：最新在前、去重、上限 10 条、不含当前."""
        os.chdir(tmp_path)
        wm = WorkspaceManager(data_dir=tmp_path / "state")
        dirs: list[Path] = []
        for i in range(12):
            d = tmp_path / f"d{i}"
            d.mkdir()
            dirs.append(d.resolve())
            wm.set_workspace(d)
        # 12 次切换后历史为最近 10 个旧 cwd：d10..d1（当前 d11 不在列）
        assert wm.recent_workspaces() == [dirs[i] for i in range(10, 0, -1)]
        assert wm.cwd not in wm.recent_workspaces()
        # limit 参数生效
        assert len(wm.recent_workspaces(limit=3)) == 3

    def test_load_filters_invalid_history_entries(self, tmp_path: Path) -> None:
        """load 恢复历史时剔除非字符串/空串/非法路径/不存在目录/重复项."""
        import json

        data_dir = tmp_path / "state"
        data_dir.mkdir()
        valid = tmp_path / "valid"
        valid.mkdir()
        payload = json.dumps(
            {
                "path": str(tmp_path / "ghost"),  # 当前路径不存在 → load 返回 None
                "history": [
                    str(valid),  # 有效保留
                    str(tmp_path / "ghost2"),  # 不存在目录 → 剔除
                    123,  # 非字符串 → 剔除
                    "   ",  # 空串 → 剔除
                    "\x00",  # resolve 抛 ValueError → 剔除
                    str(valid),  # 重复项 → 剔除
                ],
            },
            ensure_ascii=False,
        )
        (data_dir / CURRENT_WORKSPACE_FILE).write_text(payload, encoding="utf-8")
        wm = WorkspaceManager(data_dir=data_dir)
        # 当前路径无效时 load 仍返回 None，但历史中的有效项已恢复
        assert wm.load() is None
        assert wm.recent_workspaces() == [valid.resolve()]


class TestKernelCwd:
    """ReplKernel 注入 cwd / cd 命令 + 工作区事件响应."""

    def test_kernel_injects_cwd_path(self) -> None:
        """内核 namespace 中应存在 cwd（Path）."""
        bus = EventBus()
        kernel = ReplKernel(bus)
        assert "cwd" in kernel.namespace
        assert isinstance(kernel.namespace["cwd"], Path)
        assert kernel.namespace["cwd"] == Path.cwd().resolve()
        # builtin_names 包含 cwd / cd，clear() 不会清除
        assert "cwd" in kernel.builtin_names
        assert "cd" in kernel.builtin_names

    def test_kernel_cd_without_workspace_manager(self, tmp_path: Path) -> None:
        """无 WorkspaceManager 时 cd 退化为 os.chdir + 直接更新 namespace."""
        bus = EventBus()
        kernel = ReplKernel(bus)
        target = tmp_path / "d1"
        target.mkdir()
        kernel.execute(f'cd(r"{target}")')
        assert kernel.namespace["cwd"] == target.resolve()
        assert Path.cwd() == target.resolve()

    def test_kernel_cd_with_workspace_manager(self, tmp_path: Path) -> None:
        """有 WorkspaceManager 时 cd 经 WM 切换，事件总线同步更新 namespace.cwd."""
        bus = EventBus()
        wm = WorkspaceManager(bus)
        kernel = ReplKernel(bus)
        kernel.set_workspace_manager(wm)

        target = tmp_path / "d2"
        target.mkdir()
        kernel.execute(f'cd(r"{target}")')

        assert wm.cwd == target.resolve()
        assert kernel.namespace["cwd"] == target.resolve()
        assert Path.cwd() == target.resolve()

    def test_cd_invalid_path_shows_error(self, tmp_path: Path) -> None:
        """cd 不存在的目录不改变 cwd，打印错误提示."""
        bus = EventBus()
        kernel = ReplKernel(bus)
        before = kernel.namespace["cwd"]
        target = tmp_path / "nope"
        result = kernel.execute(f'cd(r"{target}")')
        assert "目录不存在" in result.stdout
        assert kernel.namespace["cwd"] == before

    def test_cd_no_arg_prints_current(self, tmp_path: Path) -> None:
        """cd() 无参数时打印当前 cwd."""
        bus = EventBus()
        wm = WorkspaceManager(bus)
        kernel = ReplKernel(bus)
        kernel.set_workspace_manager(wm)
        target = tmp_path / "d3"
        target.mkdir()
        kernel.execute(f'cd(r"{target}")')
        result = kernel.execute("cd()")
        assert str(target.resolve()) in result.stdout

    def test_clear_preserves_cwd(self) -> None:
        """clear() 清除用户变量但保留 cwd / cd / 内置符号."""
        bus = EventBus()
        kernel = ReplKernel(bus)
        kernel.execute("my_var = 42")
        assert "my_var" in kernel.namespace
        kernel.execute("clear()")
        assert "cwd" in kernel.namespace
        assert "cd" in kernel.namespace
        assert "my_var" not in kernel.namespace

    def test_workspace_change_event_updates_namespace(self, tmp_path: Path) -> None:
        """外部 WM 切换路径（不经 kernel.cd）时，内核 namespace.cwd 自动同步."""
        bus = EventBus()
        kernel = ReplKernel(bus)
        wm = WorkspaceManager(bus)
        kernel.set_workspace_manager(wm)

        target = tmp_path / "d4"
        target.mkdir()
        wm.set_workspace(target)  # 直接 WM 切换，不走 kernel.cd
        assert kernel.namespace["cwd"] == target.resolve()

    def test_restart_kernel_preserves_cwd_from_os(self, tmp_path: Path) -> None:
        """restart_kernel 重建 namespace，cwd 从 os.chdir 后的值恢复."""
        bus = EventBus()
        wm = WorkspaceManager(bus)
        kernel = ReplKernel(bus)
        kernel.set_workspace_manager(wm)
        target = tmp_path / "d5"
        target.mkdir()
        kernel.execute(f'cd(r"{target}")')
        kernel.restart_kernel()
        # _init_namespace 从 Path.cwd() 重建
        assert kernel.namespace["cwd"] == Path.cwd().resolve()
        assert isinstance(kernel.namespace["cwd"], Path)


class TestWorkspaceHistoryLimitConfigurable:
    """WorkspaceManager 历史条数上限可经 SettingsPanel 调整."""

    def test_history_limit_follows_runtime_config(self, tmp_path: Path) -> None:
        """连续切换超过当前 runtime history_limit 条数时，多出的应被截断."""
        bus = EventBus()
        wm = WorkspaceManager(bus)

        # 把 runtime limit 改成 3
        update_runtime_config(workspace_history_limit=3)
        assert get_workspace_history_limit() == 3

        # 连续切换 4 次（每条都不同）
        for i in range(4):
            target = tmp_path / f"d{i}"
            target.mkdir()
            wm.set_workspace(target)

        # 历史应被截断到 3 条（最新的 3 次切换前路径）
        assert len(wm._history) <= 3
        # 当前 cwd 不在历史中
        assert wm.cwd not in wm._history

    def test_history_limit_change_applies_to_existing_manager(self, tmp_path: Path) -> None:
        """runtime limit 在运行中被收紧，下次切换时历史会被裁剪."""
        bus = EventBus()
        wm = WorkspaceManager(bus)

        # 默认 limit=10，切换 4 次
        for i in range(4):
            target = tmp_path / f"d{i}"
            target.mkdir()
            wm.set_workspace(target)

        assert len(wm._history) == 4

        # 把 limit 收紧到 2——下次切换时才生效（_update_history 里裁剪）
        update_runtime_config(workspace_history_limit=2)
        target5 = tmp_path / "d4"
        target5.mkdir()
        wm.set_workspace(target5)

        assert len(wm._history) == 2

    def test_save_persists_truncated_history(self, tmp_path: Path, monkeypatch) -> None:
        """save() 持久化的历史条数也应遵守 runtime limit."""
        bus = EventBus()
        data_dir = tmp_path / "fake_data"
        data_dir.mkdir()  # save() 需要目录存在
        monkeypatch.setattr("zylab.sci.workspace.default_data_dir", lambda: data_dir)
        wm = WorkspaceManager(bus)

        # 切换 3 次
        for i in range(3):
            target = tmp_path / f"d{i}"
            target.mkdir()
            wm.set_workspace(target)

        # limit 收紧到 2
        update_runtime_config(workspace_history_limit=2)
        wm.save()

        # 从持久化文件验证
        import json

        persisted = json.loads((data_dir / CURRENT_WORKSPACE_FILE).read_text(encoding="utf-8"))
        assert "history" in persisted
        # 持久化历史 <= 运行时值（save 前会过滤掉当前路径）
        assert len(persisted["history"]) <= 2

    def test_load_respects_runtime_limit(self, tmp_path: Path, monkeypatch) -> None:
        """从持久化文件恢复时，历史条目也应被 runtime limit 裁剪."""
        bus = EventBus()
        data_dir = tmp_path / "fake_data"
        data_dir.mkdir()
        monkeypatch.setattr("zylab.sci.workspace.default_data_dir", lambda: data_dir)

        # 先准备一个含 5 条历史的 workspace.json
        import json

        history_paths = []
        for i in range(5):
            p = tmp_path / f"d{i}"
            p.mkdir()
            history_paths.append(str(p))
        target_current = tmp_path / "current"
        target_current.mkdir()

        (data_dir / CURRENT_WORKSPACE_FILE).write_text(
            json.dumps({"path": str(target_current), "history": history_paths}, ensure_ascii=False),
            encoding="utf-8",
        )

        # runtime limit 收紧到 3
        update_runtime_config(workspace_history_limit=3)

        # 新建 WM，load 应裁剪历史
        wm = WorkspaceManager(bus)
        wm.load()
        assert len(wm._history) <= 3
        assert wm.cwd == target_current.resolve()
