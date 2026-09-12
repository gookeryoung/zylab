## 目标

在**仅动 matplotlib REPL 全局样式**、保持**精致网格风格**(whitegrid 基础 + 浅色细化网格、曲线更突出)的前提下,深化 `src/zylab/sci/plotting.py` 的 `_MPL_RC_OVERRIDES`,让用户 `plt.plot(...)` 出的曲线图开箱即得精致的显示效果。不改 pyqtgraph GUI 渲染、不改报告 SVG 路径、不新增辅助函数。

## 改动内容

### 1. `src/zylab/sci/plotting.py` — 扩充 `_MPL_RC_OVERRIDES`(L61-74)

在现有条目基础上按分组补充(全部为曲线图相关细节):

- **网格精致化**:`grid.color` 浅灰(如 `#d8d8d8`)、`grid.linestyle = ":"`、`grid.linewidth = 0.8`、`grid.alpha = 0.7`;`axes.grid.axis = "both"`,`axes.grid.which = "major"`;`axes.axisbelow = True` 保证网格在曲线下层。
- **坐标轴**: `axes.edgecolor` 浅灰、`axes.linewidth = 0.8`(细边框衬托曲线)、`axes.labelsize = "medium"`、`axes.titlesize = "large"` + `axes.titleweight = "semibold"`。
- **刻度**: `xtick/ytick.labelsize = "small"`、`xtick/ytick.direction = "out"`、`xtick/ytick.major.width = 0.8`、`xtick/ytick.minor.visible = True`(对数轴/细节查看更友好)。
- **曲线**: `lines.antialiased = True`、`lines.solid_capstyle = "round"`(曲线端点圆润)、`lines.markeredgewidth = 1.0`。
- **图例**: `legend.edgecolor` 浅灰、`legend.fancybox = True`、`legend.borderaxespad = 0.8`。
- **导出**: `savefig.bbox = "tight"`、`savefig.facecolor = "white"`、`figure.facecolor = "white"`。

同步更新模块 docstring(L3-13)和 `_MPL_RC_OVERRIDES` 的注释,说明 whitegrid 已接管与 zylab 覆盖项的分工。

### 2. `tests/test_sci_plotting.py` — 新增/调整断言

- 保留并复用现有测试结构,新增 1-2 个测试:如 `test_apply_defaults_refined_grid`(断言 `grid.linestyle == ":"`、`grid.alpha > 0`、`axes.axisbelow is True`)与 `test_apply_defaults_tick_and_legend`(断言刻度方向、`legend.fancybox`、`savefig.bbox == "tight"`)。
- 现有断言(`axes.grid` True、`lines.linewidth` 2.0 等)不受影响,无需改动。

### 3. 验证

- 运行 `python -m pytest tests/test_sci_plotting.py -x -q` 确认全部通过。
- 幂等性由现有 `test_apply_defaults_idempotent` 覆盖,新增键自动纳入。

## 不做的事

- 不改 `palettes.py` 色板、不动 seaborn 主题选择(保持 `whitegrid`/`notebook`)。
- 不改 pyqtgraph(`notebook_page.py`、`dsl_result_view.py`)与报告 SVG(`flowchart/report.py`)。
- 不新增公共 API(无 `curve_style()`、无 `export_figure()`)。