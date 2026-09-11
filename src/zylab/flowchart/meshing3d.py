"""三维 HEX8 结构化网格生成器：圆柱电阻与 V 形薄膜电阻（陶瓷基底）.

两类电阻模型均以 HEX8 六面体单元离散（Mesh 层保证 3D 网格维度合法）：

- 圆柱电阻：极坐标结构化网格（周向闭合、径向分层、轴向拉伸），轴心留
  5% 半径中心孔（HEX8 退化到轴心会雅可比奇异，留小孔近似实心圆柱）；
- V 形薄膜电阻：俯视 V 形路径扫掠（薄膜厚度微米级），截面法向取相邻
  弦向的平均方向保证折点处截面连续；陶瓷基底与薄膜共享底面节点，
  实现跨材料热耦合（电学上陶瓷电导率极小，装配时被绝缘过滤跳过）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zylab.fea import ElementBlock, ElementType, Mesh, MeshError

__all__ = [
    "CylinderMesh3d",
    "VfilmMesh3d",
    "cylinder_resistor_mesh",
    "vfilm_resistor_mesh",
]

#: 四边形面片节点组（对流边界口径，与 fea.thermal.Convection.faces 对齐）
Face = tuple[int, int, int, int]

#: 圆柱轴心中心孔半径比（HEX8 无法退化到轴心，留小孔近似实心）
_INNER_RATIO = 0.05

#: 坐标合并容差的十进制位数（1e-9 mm，薄膜 μm 级特征尺寸远大于此）
_ROUND_DIGITS = 9


@dataclass(frozen=True)
class CylinderMesh3d:
    """圆柱电阻网格与边界引用.

    Attributes:
        mesh: HEX8 网格（单一电阻材料块）。
        end_low_nodes: z=0 端面节点（接地 0V 电极）。
        end_high_nodes: z=L 端面节点（给定电压电极）。
        conv_faces: 外圆柱面与两端面的四边形面片（对流边界）。
    """

    mesh: Mesh
    end_low_nodes: tuple[int, ...]
    end_high_nodes: tuple[int, ...]
    conv_faces: tuple[Face, ...]


def cylinder_resistor_mesh(
    radius: float,
    length: float,
    n_theta: int,
    n_r: int,
    n_z: int,
) -> CylinderMesh3d:
    """生成圆柱电阻 HEX8 网格（极坐标结构化、周向闭合）.

    节点编号 ``idx(i_z, i_r, i_th) = i_z*(n_r+1)*n_theta + i_r*n_theta + i_th``，
    周向 ``i_th`` 以 ``n_theta`` 取模闭合；单元节点序按自然坐标
    (ξ=r, η=θ, ζ=z) 排列，柱坐标右旋保证雅可比行列式恒正。

    Args:
        radius: 圆柱半径 R（mm，> 0）。
        length: 圆柱长度 L（mm，> 0）。
        n_theta: 周向分段数（≥ 3）。
        n_r: 径向分段数（≥ 1）。
        n_z: 轴向分段数（≥ 1）。

    Returns:
        网格与端面电极节点、表面对流面片。

    Raises:
        MeshError: 几何或分段参数非法。
    """
    if radius <= 0.0 or length <= 0.0:
        raise MeshError(f"圆柱半径与长度须为正，实际 R={radius}, L={length}")
    if n_theta < 3 or n_r < 1 or n_z < 1:
        raise MeshError(f"分段数非法（周向≥3、径向/轴向≥1），实际 n_theta={n_theta}, n_r={n_r}, n_z={n_z}")

    r_in = radius * _INNER_RATIO
    radii = np.linspace(r_in, radius, n_r + 1)
    d_theta = 2.0 * np.pi / n_theta
    zs = np.linspace(0.0, length, n_z + 1)

    # 节点坐标：按 (i_z, i_r, i_th) C 序展平，与 idx 公式一致
    # （meshgrid(indexing="ij") 形状 (n_theta, n_r+1, n_z+1) → 转置到 (i_z, i_r, i_th)）
    theta = np.arange(n_theta) * d_theta
    grid_th, grid_r, grid_z = np.meshgrid(theta, radii, zs, indexing="ij")
    stacked = np.stack((grid_r * np.cos(grid_th), grid_r * np.sin(grid_th), grid_z), axis=-1)
    coords = stacked.transpose(2, 1, 0, 3).reshape(-1, 3)

    def idx(i_z: int, i_r: int, i_th: int) -> int:
        """节点编号（周向闭合取模）."""
        return i_z * (n_r + 1) * n_theta + i_r * n_theta + (i_th % n_theta)

    conn = np.empty((n_r * n_theta * n_z, 8), dtype=np.int64)
    row = 0
    for i_z in range(n_z):
        for i_r in range(n_r):
            for i_th in range(n_theta):
                j = i_th + 1
                lo = (idx(i_z, i_r, i_th), idx(i_z, i_r + 1, i_th), idx(i_z, i_r + 1, j), idx(i_z, i_r, j))
                hi = (
                    idx(i_z + 1, i_r, i_th),
                    idx(i_z + 1, i_r + 1, i_th),
                    idx(i_z + 1, i_r + 1, j),
                    idx(i_z + 1, i_r, j),
                )
                conn[row] = (*lo, *hi)
                row += 1
    mesh = Mesh(coords=coords, blocks=(ElementBlock(etype=ElementType.HEX8, conn=conn, name="电阻体"),))

    # 外圆柱面（r = R）与两端面的对流面片
    faces: list[Face] = []
    for i_z in range(n_z):
        for i_th in range(n_theta):
            j = i_th + 1
            faces.append((idx(i_z, n_r, i_th), idx(i_z, n_r, j), idx(i_z + 1, n_r, j), idx(i_z + 1, n_r, i_th)))
    for i_r in range(n_r):
        for i_th in range(n_theta):
            j = i_th + 1
            for k in (0, n_z):
                faces.append((idx(k, i_r, i_th), idx(k, i_r, j), idx(k, i_r + 1, j), idx(k, i_r + 1, i_th)))

    n_ring = (n_r + 1) * n_theta
    return CylinderMesh3d(
        mesh=mesh,
        end_low_nodes=tuple(range(n_ring)),
        end_high_nodes=tuple(range(n_z * n_ring, (n_z + 1) * n_ring)),
        conv_faces=tuple(faces),
    )


@dataclass(frozen=True)
class VfilmMesh3d:
    """V 形薄膜电阻网格与边界引用.

    Attributes:
        mesh: HEX8 网格（电阻膜 / 电极 / 陶瓷基底三块，材料索引 0/1/2）。
        lead_low_nodes: 引入端面节点（x=0，接地 0V 电极）。
        lead_high_nodes: 引出端面节点（x=L，给定电压电极）。
        film_top_faces: 薄膜顶面（z = 膜厚）四边形面片（对流边界）。
        base_bottom_nodes: 陶瓷基底底面（z = -基底厚）节点（恒温边界）。
    """

    mesh: Mesh
    lead_low_nodes: tuple[int, ...]
    lead_high_nodes: tuple[int, ...]
    film_top_faces: tuple[Face, ...]
    base_bottom_nodes: tuple[int, ...]


class _NodeMap:
    """坐标去重的节点登记表（相同坐标返回同一节点号）.

    段间共享站点与薄膜/基底共享底面均靠坐标一致自动合并。
    """

    def __init__(self) -> None:
        self._index: dict[tuple[float, float, float], int] = {}
        self._coords: list[list[float]] = []

    def node(self, x: float, y: float, z: float) -> int:
        """登记坐标并返回节点号（已存在则复用）."""
        key = (round(x, _ROUND_DIGITS), round(y, _ROUND_DIGITS), round(z, _ROUND_DIGITS))
        found = self._index.get(key)
        if found is not None:
            return found
        new_id = len(self._coords)
        self._coords.append([key[0], key[1], key[2]])
        self._index[key] = new_id
        return new_id

    def coords(self) -> np.ndarray:
        """全部节点坐标表."""
        return np.asarray(self._coords, dtype=np.float64)


def vfilm_resistor_mesh(  # noqa: PLR0912, PLR0913, PLR0917  几何十参数各自独立（跨/深/宽/厚/基底/电极 + 四向剖分），三块网格生成分支不可合并
    span: float,
    depth: float,
    width: float,
    thickness: float,
    substrate_h: float,
    lead_len: float,
    n_lead: int,
    n_diag: int,
    n_width: int,
    n_sub: int,
) -> VfilmMesh3d:
    """生成 V 形薄膜电阻 HEX8 网格（俯视 V 形路径扫掠 + 陶瓷基底拉伸）.

    俯视路径为四段折线：引入段（电极）→ 左斜段 → 右斜段 → 引出段（电极），
    V 顶点下探 ``depth``；每站截面（宽 ``width`` × 厚 ``thickness``）沿路径
    扫掠一层薄膜单元，截面法向取相邻弦向平均（折点处角平分），保证段间
    共享闭合。陶瓷基底与薄膜同宽，从薄膜底面向下拉伸 ``n_sub`` 层，共享
    z=0 节点实现热耦合。

    Args:
        span: 总跨度 L（mm，> 2*lead_len）。
        depth: V 形深度 d（mm，> 0）。
        width: 薄膜宽度 w（mm，> 0）。
        thickness: 薄膜厚度（mm，> 0，典型微米级如 0.005）。
        substrate_h: 陶瓷基底厚度（mm，> 0）。
        lead_len: 引入/引出电极段长 a（mm，> 0，< span/2）。
        n_lead: 每段电极的单元数（≥ 1）。
        n_diag: 每段斜线的单元数（≥ 1）。
        n_width: 宽度方向分段数（≥ 1）。
        n_sub: 基底厚度方向层数（≥ 1）。

    Returns:
        网格与电极端面节点、薄膜顶面对流面片、基底底面恒温节点。

    Raises:
        MeshError: 几何或分段参数非法。
    """
    if min(span, depth, width, thickness, substrate_h, lead_len) <= 0.0:
        raise MeshError("V 形薄膜几何参数（跨度/深度/宽/厚/基底厚/电极段长）须为正")
    if 2.0 * lead_len >= span:
        raise MeshError(f"电极段过长（2a={2.0 * lead_len} ≥ 跨度 L={span}），须留出 V 形区")
    if min(n_lead, n_diag, n_width, n_sub) < 1:
        raise MeshError(f"分段数须 ≥ 1，实际 n_lead={n_lead}, n_diag={n_diag}, n_width={n_width}, n_sub={n_sub}")

    # 俯视折线顶点与各段单元数：引入 → 左斜 → 右斜 → 引出
    vertices = (
        (0.0, 0.0),
        (lead_len, 0.0),
        (span / 2.0, -depth),
        (span - lead_len, 0.0),
        (span, 0.0),
    )
    seg_counts = (n_lead, n_diag, n_diag, n_lead)

    # 站点序列（相邻段共享端点站）
    stations: list[tuple[float, float]] = []
    for (x0, y0), (x1, y1), count in zip(vertices[:-1], vertices[1:], seg_counts):
        for i in range(count):
            stations.append((x0 + (x1 - x0) * i / count, y0 + (y1 - y0) * i / count))
    stations.append(vertices[-1])
    n_st = len(stations)  # 站点数 = 2*n_lead + 2*n_diag + 1

    # 各站截面法向：相邻弦向的平均（折点角平分，首末站取单侧）
    chords = []
    for k in range(n_st - 1):
        dx = stations[k + 1][0] - stations[k][0]
        dy = stations[k + 1][1] - stations[k][1]
        norm = (dx * dx + dy * dy) ** 0.5
        chords.append((dx / norm, dy / norm))
    normals: list[tuple[float, float]] = []
    for i in range(n_st):
        prev = chords[max(i - 1, 0)]
        nxt = chords[min(i, n_st - 2)]
        vx, vy = prev[0] + nxt[0], prev[1] + nxt[1]
        norm = (vx * vx + vy * vy) ** 0.5
        if norm < 1.0e-12:
            raise MeshError("路径折返 180°，截面法向无法定义")
        # e_z × u（弦向 u 绕 z 轴转 90°），垂直于路径的宽度方向
        normals.append((-vy / norm, vx / norm))

    mapper = _NodeMap()

    # 薄膜截面节点：站 i × 宽 j，底 z=0 / 顶 z=thickness
    bot: list[list[int]] = []
    top: list[list[int]] = []
    for i, (px, py) in enumerate(stations):
        nx, ny = normals[i]
        bot_row: list[int] = []
        top_row: list[int] = []
        for j in range(n_width + 1):
            offset = width * (j / n_width - 0.5)
            qx, qy = px + nx * offset, py + ny * offset
            bot_row.append(mapper.node(qx, qy, 0.0))
            top_row.append(mapper.node(qx, qy, thickness))
        bot.append(bot_row)
        top.append(top_row)

    # 基底截面节点：站 i × 宽 j × 层 m（m=0 与薄膜底面共享），z = -H*m/n_sub
    sub: list[list[list[int]]] = []
    for i, (px, py) in enumerate(stations):
        nx, ny = normals[i]
        rows: list[list[int]] = []
        for m in range(n_sub + 1):
            z = -substrate_h * m / n_sub
            row: list[int] = []
            for j in range(n_width + 1):
                offset = width * (j / n_width - 0.5)
                row.append(mapper.node(px + nx * offset, py + ny * offset, z))
            rows.append(row)
        sub.append(rows)

    # 薄膜单元：底面 (k,j),(k+1,j),(k+1,j+1),(k,j+1) + 顶面同序
    # （自然坐标 ξ=路径向、η=宽度向、ζ=厚度向，(ξ×η)·ζ > 0 保证雅可比正）
    def _hex(bottom: tuple[int, int, int, int], upper: tuple[int, int, int, int]) -> tuple[int, ...]:
        return (*bottom, *upper)

    conn_resistor: list[tuple[int, ...]] = []
    conn_electrode: list[tuple[int, ...]] = []
    n_resistor_begin = n_lead
    n_resistor_end = n_lead + 2 * n_diag
    for k in range(n_st - 1):
        for j in range(n_width):
            cell = _hex(
                (bot[k][j], bot[k + 1][j], bot[k + 1][j + 1], bot[k][j + 1]),
                (top[k][j], top[k + 1][j], top[k + 1][j + 1], top[k][j + 1]),
            )
            if n_resistor_begin <= k < n_resistor_end:
                conn_resistor.append(cell)
            else:
                conn_electrode.append(cell)

    # 基底单元：层 m（ζ=-1 取低层 m+1，ζ=+1 取高层 m；sub 索引序 [站][层][宽]）
    conn_substrate: list[tuple[int, ...]] = []
    for k in range(n_st - 1):
        for j in range(n_width):
            for m in range(n_sub):
                conn_substrate.append(
                    _hex(
                        (sub[k][m + 1][j], sub[k + 1][m + 1][j], sub[k + 1][m + 1][j + 1], sub[k][m + 1][j + 1]),
                        (sub[k][m][j], sub[k + 1][m][j], sub[k + 1][m][j + 1], sub[k][m][j + 1]),
                    )
                )

    mesh = Mesh(
        coords=mapper.coords(),
        blocks=(
            ElementBlock(
                etype=ElementType.HEX8, conn=np.asarray(conn_resistor, dtype=np.int64), material=0, name="电阻膜"
            ),
            ElementBlock(
                etype=ElementType.HEX8, conn=np.asarray(conn_electrode, dtype=np.int64), material=1, name="电极"
            ),
            ElementBlock(
                etype=ElementType.HEX8, conn=np.asarray(conn_substrate, dtype=np.int64), material=2, name="陶瓷基底"
            ),
        ),
    )

    # 薄膜顶面对流面片与边界节点
    film_faces = [
        (top[k][j], top[k + 1][j], top[k + 1][j + 1], top[k][j + 1]) for k in range(n_st - 1) for j in range(n_width)
    ]
    lead_low = (*bot[0], *top[0])
    lead_high = (*bot[-1], *top[-1])
    base_bottom = tuple(sub[i][n_sub][j] for i in range(n_st) for j in range(n_width + 1))
    return VfilmMesh3d(
        mesh=mesh,
        lead_low_nodes=lead_low,
        lead_high_nodes=lead_high,
        film_top_faces=tuple(film_faces),
        base_bottom_nodes=base_bottom,
    )
