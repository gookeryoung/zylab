"""临时脚本：把 MainWindow 里已迁移到 AppController 的方法体替换为转发调用."""

from __future__ import annotations

import re
from pathlib import Path

path = Path("src/zylab/gui/main_window.py")
text = path.read_text(encoding="utf-8")

# 逐个方法替换
replacements: list[tuple[str, str]] = []

# 1. _on_switch_workspace
old = (
    "    def _on_switch_workspace(self) -> None:\n"
    '        """弹出目录选择对话框，确认后经 WorkspaceManager 切换工作区."""\n'
    "        current = str(self._workspace_manager.cwd)\n"
    "        target = QFileDialog.getExistingDirectory(\n"
    "            self,\n"
    '            "选择工作区目录",\n'
    "            current,\n"
    "            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,\n"
    "        )\n"
    "        if not target:\n"
    "            return  # 用户取消\n"
    "        self._switch_workspace_to(target)\n"
)
new = (
    "    def _on_switch_workspace(self) -> None:\n"
    '        """弹出目录选择对话框（转发给 AppController）."""\n'
    "        self._controller._on_pick_workspace()\n"
)
replacements.append((old, new))

# 2. _refresh_and_show_workspace_menu + _refresh_workspace_menu（连续两个方法）
old = (
    "    def _refresh_and_show_workspace_menu(self) -> None:\n"
    '        """刷新历史下拉菜单并立即弹出（供按钮点击或 setMenu 自动触发）."""\n'
    "        self._refresh_workspace_menu()\n"
    "\n"
    "    def _refresh_workspace_menu(self) -> None:\n"
    '        """重建历史工作区菜单（最近 10 条，点击即切换）."""\n'
    "        menu = self._workspace_history_menu\n"
    "        menu.clear()\n"
    "        history = self._workspace_manager.recent_workspaces(limit=10)\n"
    "        if not history:\n"
    "            # 无历史：显示禁用占位项\n"
    '            empty = menu.addAction("（暂无历史）")\n'
    "            empty.setEnabled(False)\n"
    "            return\n"
    "        for path in history:\n"
    "            action = menu.addAction(str(path))\n"
    "            action.setData(str(path))\n"
    "            action.triggered.connect(lambda _checked=False, p=str(path): self._switch_workspace_to(p))\n"
    "        menu.addSeparator()\n"
    '        open_action = menu.addAction("选择其他目录…")\n'
    "        open_action.triggered.connect(self._on_switch_workspace)\n"
)
new = (
    "    def _refresh_and_show_workspace_menu(self) -> None:\n"
    '        """刷新历史下拉菜单并立即弹出（转发给 AppController）."""\n'
    "        self._controller._refresh_workspace_menu()\n"
    "\n"
    "    def _refresh_workspace_menu(self) -> None:\n"
    '        """重建历史工作区菜单（转发给 AppController）."""\n'
    "        self._controller._refresh_workspace_menu()\n"
)
replacements.append((old, new))

# 3. _switch_workspace_to — 读一下再替换
# 这方法体较长，后面单独处理

# 4. _open_about_dialog
old_about_start = "    def _open_about_dialog(self) -> None:"
idx_about = text.find(old_about_start)
# 找到下一个顶层方法（非缩进）
next_def = text.find("\n    def ", idx_about + 1)
if next_def == -1:
    next_def = len(text)
old_about = text[idx_about:next_def]
new_about = (
    "    def _open_about_dialog(self) -> None:\n"
    '        """弹出关于对话框（转发给 AppController）."""\n'
    "        self._controller.open_about_dialog()\n"
)
replacements.append((old_about, new_about))

# 5. _open_settings_dialog
old_settings_start = "    def _open_settings_dialog(self) -> None:"
idx_settings = text.find(old_settings_start)
next_def2 = text.find("\n    def ", idx_settings + 1)
if next_def2 == -1:
    next_def2 = len(text)
old_settings = text[idx_settings:next_def2]
new_settings = (
    "    def _open_settings_dialog(self) -> None:\n"
    '        """弹出设置对话框（转发给 AppController）."""\n'
    "        self._controller.open_settings_dialog()\n"
)
replacements.append((old_settings, new_settings))

# 6. set_run_status
old_rs_start = '    def set_run_status(self, state: str, detail: str = "") -> None:'
idx_rs = text.find(old_rs_start)
# set_run_status 是最后一个方法，直接到文件末尾
old_rs = text[idx_rs:]
new_rs = (
    '    def set_run_status(self, state: str, detail: str = "") -> None:\n'
    '        """设置右下角运行状态 indicator（转发给 AppController）."""\n'
    "        self._controller.set_run_status(state, detail)\n"
)
replacements.append((old_rs, new_rs))

# 执行替换
for old_t, new_t in replacements:
    if old_t not in text:
        print(f"⚠️  未找到: {old_t[:60]}...")
        continue
    text = text.replace(old_t, new_t, 1)
    print(f"✅ 替换完成 ({len(old_t)} chars -> {len(new_t)} chars)")

# 7. _switch_workspace_to — 单独处理（用正则）
pattern_sw = re.compile(
    r"    def _switch_workspace_to\(self, target: str\) -> None:.*?(?=\n    def )",
    re.DOTALL,
)
new_sw = (
    "    def _switch_workspace_to(self, target: str) -> None:\n"
    '        """切换到指定工作区路径（转发给 AppController）."""\n'
    "        self._controller.switch_workspace_to(target)\n"
    "\n"
    "    def _on_switch_workspace_placeholder(self) -> None:\n"
    "        pass\n"
)
# 先匹配替换
match = pattern_sw.search(text)
if match:
    # 替换时保留后面的 \n    def 前缀
    after = text[match.end() :]
    text = text[: match.start()] + new_sw.rstrip() + after
    # 删除我加的 placeholder
    text = text.replace(
        "    def _on_switch_workspace_placeholder(self) -> None:\n        pass\n\n",
        "",
    )
    print("✅ _switch_workspace_to 替换完成")
else:
    print("⚠️  未找到 _switch_workspace_to")

path.write_text(text, encoding="utf-8")
print(f"\n📝 写入 {path}，共 {text.count(chr(10)) + 1} 行")
