"""扫描 SVG 图标资源，生成 ``resources.qrc`` 并编译为 ``resources_rc.py``。

将 SVG 图标打包进 Qt 资源系统（qrc），运行时通过 ``qrc:/icons/`` 路径访问，
减少磁盘 I/O，加快启动速度。替代原方案中运行时生成临时 SVG 的机制。

使用方式::

    uv run python scripts/build_qrc.py

输出：

- ``src/zylab/gui/resources.qrc``：资源清单（XML）
- ``src/zylab/gui/resources_rc.py``：编译后的 Python 模块，供 ``app.py`` import

代码内引用规则：

- Python 代码：``from zylab.gui import resources_rc  # noqa: F401`` 注册资源
- QSS 样式表：QIcon("qrc:/icons/xxx.svg")
- 图标令牌常量：``zylab.gui.theme.ICONS_PREFIX = "qrc:/icons/"``
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

# 仓库根目录（scripts/build_qrc.py 的上两级）
ROOT = Path(__file__).resolve().parent.parent
# 图标目录
ICONS_DIR = ROOT / "src" / "zylab" / "assets" / "icons"
# 输出文件
QRC_FILE = ROOT / "src" / "zylab" / "gui" / "resources.qrc"
RC_FILE = ROOT / "src" / "zylab" / "gui" / "resources_rc.py"

__all__ = ["main"]


def collect_icon_files() -> list[tuple[str, Path]]:
    """收集所有 .svg 图标，返回 (qrc_alias, abs_path) 列表.

    :return: alias 形如 ``icons/home.svg``，对应 qrc 内路径 ``qrc:/icons/home.svg``
    """
    files: list[tuple[str, Path]] = []
    for svg in sorted(ICONS_DIR.glob("*.svg")):
        alias = f"icons/{svg.name}"
        files.append((alias, svg))
    return files


def write_qrc(icon_files: list[tuple[str, Path]]) -> None:
    """生成 .qrc 文件.

    :param icon_files: SVG 文件 (alias, abs_path) 列表
    """
    rcc = ET.Element("RCC")
    rcc.set("version", "1.0")
    qresource = ET.SubElement(rcc, "qresource", {"prefix": "/"})
    for alias, path in icon_files:
        # qrc 内 <file> 路径相对于 .qrc 文件所在目录解析
        rel = os.path.relpath(path, QRC_FILE.parent).replace("\\", "/")
        ET.SubElement(qresource, "file", {"alias": alias}).text = rel
    raw = ET.tostring(rcc, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ", encoding="utf-8")
    pretty_str = pretty.decode("utf-8")
    lines = [line for line in pretty_str.splitlines() if line.strip()]
    QRC_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def detect_rcc_tool() -> str:
    """检测 pyside2-rcc 或 pyside6-rcc 命令.

    优先使用 PATH 中的 ``pyside2-rcc``/``pyside6-rcc``；找不到则回退到
    ``python -m`` 调用方式。

    :return: 可用的 rcc 命令名
    :raises RuntimeError: 两个工具都不可用
    """
    for tool in ("pyside2-rcc", "pyside6-rcc"):
        if shutil.which(tool):
            return tool
    try:
        import PySide2  # noqa: F401

        return "pyside2-rcc"
    except ImportError:
        pass
    try:
        import PySide6  # noqa: F401

        return "pyside6-rcc"
    except ImportError:
        pass
    raise RuntimeError("未找到 pyside2-rcc 或 pyside6-rcc，请安装 PySide2 或 PySide6")


def compile_qrc() -> None:
    """调用 pyside2-rcc 编译 .qrc 为 resources_rc.py."""
    tool = detect_rcc_tool()
    cmd = [tool, "-o", str(RC_FILE), str(QRC_FILE)]
    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0 or not RC_FILE.exists():
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"rcc 编译失败（退出码 {result.returncode}）")
    print(f"编译产物: {RC_FILE.relative_to(ROOT)} ({RC_FILE.stat().st_size} bytes)")


def main() -> int:
    """入口函数.

    :return: 退出码（0 成功）
    """
    icon_files = collect_icon_files()
    print(f"收集 SVG 图标 {len(icon_files)} 个")

    write_qrc(icon_files)
    print(f"生成清单: {QRC_FILE.relative_to(ROOT)}")

    compile_qrc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
