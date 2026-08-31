"""内置分析模板：随包发布的预连节点图（assets/templates/<学科>/*.json）.

预制模板以人类可读 JSON 分发并按学科子目录归类；用户可参照其结构
在 ``data_dir/templates/<学科>/`` 下自建模板（注册表递归加载），
也可直接修改包内 JSON 定制预制模板的行为。
"""

from __future__ import annotations

import logging
from pathlib import Path

from .errors import TemplateError
from .template import Template, template_from_json

__all__ = ["BUILTIN_TEMPLATES", "builtin_templates_dir"]

logger = logging.getLogger(__name__)

#: 预制模板资产目录（随包分发，按学科子目录归类）
_ASSETS_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "assets" / "templates"


def builtin_templates_dir() -> Path:
    """预制模板资产目录（供打包校验与用户查阅）."""
    return _ASSETS_TEMPLATES_DIR


def _load_assets_templates(directory: Path) -> tuple[Template, ...]:
    """加载目录下全部 ``<学科>/*.json`` 预制模板.

    单个文件非法仅记录告警并跳过（缺目录视为打包缺陷，返回空并告警）。
    """
    if not directory.is_dir():
        logger.warning("预制模板目录缺失: %s", directory)
        return ()
    templates: list[Template] = []
    for path in sorted(directory.glob("*/*.json")):
        try:
            templates.append(template_from_json(path.read_text(encoding="utf-8")))
        except (OSError, TemplateError) as exc:
            logger.warning("跳过非法预制模板 %s: %s", path.name, exc)
    return tuple(templates)


#: 内置模板表（模块导入时从包内 JSON 资产加载）
BUILTIN_TEMPLATES: tuple[Template, ...] = _load_assets_templates(_ASSETS_TEMPLATES_DIR)
