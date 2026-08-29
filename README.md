# zylab

> 通用科学计算仿真分析平台。

[![PyPI](https://img.shields.io/pypi/v/zylab)](https://pypi.org/project/zylab/)
[![CI](https://github.com/gookeryoung/zylab/actions/workflows/ci.yml/badge.svg)](https://github.com/gookeryoung/zylab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Coverage](https://img.shields.io/badge/coverage-%E2%89%A595%25-brightgreen.svg)

## 特性

- **构建工具链**：hatchling + uv + ruff + pyrefly + pytest + coverage
- **Python 版本**：3.11 ~ 3.14
- **代码质量**：pre-commit 钩子 + ruff lint/format，覆盖率阈值 95%
- **CI/CD**：GitHub Actions（lint + typecheck + 多版本测试 + 自动发布到 PyPI）
- **文档**：Sphinx + ReadTheDocs（中文 zh_CN）
- **多版本测试**：tox + tox-uv（py311, py312, py313, py314）
- **容器化**：Dockerfile（含国内镜像源配置）
- **GUI 框架**：PySide2（Qt5，Python≤3.10）/ PySide6（Qt6，Python≥3.11），按版本自动区分
- **项目结构**：src layout + py.typed 类型标记

## 安装

```bash
pip install zylab
```

或使用 [uv](https://docs.astral.sh/uv/)：

```bash
uv add zylab
```

## 快速上手

```python
import zylab

print(zylab.__version__)
```

## 开发

```bash
# 安装开发依赖
uv sync --extra dev

# 运行测试（含覆盖率，阈值 95%）
uv run pytest -m "not slow" --cov=zylab --cov-fail-under=95

# 类型检查
uv run pyrefly check .

# 代码风格
uv run ruff check src tests
uv run ruff format --check src tests
```

### Make 快捷命令

项目提供 Makefile 封装常用操作，运行 `make help` 查看全部命令：

```bash
make sync     # 安装开发依赖
make check    # 全套门禁 (lint + typecheck + cov)
make build    # 构建分发包
make clean    # 清理构建产物
make bump PART=patch  # 版本号 bump
```


## 文档

文档由 Sphinx 构建，托管在 ReadTheDocs：

```bash
# 本地构建文档
make doc
```


## 多版本测试

使用 tox 在多个 Python 版本（py311, py312, py313, py314）下运行测试：

```bash
make tox
```

## 许可证

MIT
