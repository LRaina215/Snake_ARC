# -*- coding: utf-8 -*-
"""圆弧滚动运动（Rolling Helix-Arc）研究路线 §2 主要解决问题 · 示意图绘制。

绘制风格沿用 Snake_Robot_Gait 项目的 matplotlib 示意风格：
半透明灰蓝管道 + 蓝/橙/紫三色机器人分段 + ①②③ 阶段编号 + 面板注释 + 底部图例，
并在副标题中注明"机理/过程示意，非仿真与实机结果"。
"""
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Ellipse

HERE = Path(__file__).resolve().parent
FONT = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': FONT.get_name(), 'axes.unicode_minus': False})

BLUE, ORANGE, PURPLE = '#176eab', '#ee9c24', '#8458ac'   # 使能 / 过渡 / 失能
PIPE, DASH, CAP = '#8ea3ab', '#8c9ba1', '#dbe2e5'        # 管道填充 / 中心虚线 / 端盖
TITLE, SUB, NOTE = '#17354a', '#596b77', '#3e5666'


# ---------------------------------------------------------------- 基础图元
def qbez(p0, pc, p1, n=90):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.asarray(p0, float) + 2 * (1 - t) * t * np.asarray(pc, float) + t ** 2 * np.asarray(p1, float)


def arc_pts(c, r, a0, a1, n=140):
    t = np.radians(np.linspace(a0, a1, n))
    return np.column_stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)])


def _normals(pts):
    d = np.gradient(pts, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    L[L == 0] = 1
    return np.stack([-d[:, 1] / L, d[:, 0] / L], 1)


def _endcap(ax, p, tangent, hw, z):
    ang = math.degrees(math.atan2(tangent[1], tangent[0])) + 90
    ax.add_patch(Ellipse(tuple(p), .5, 2 * hw, angle=ang, facecolor=CAP, edgecolor=PIPE, lw=.9, zorder=z + .4))


def curved_band(ax, pts, hw, cap_start=False, cap_end=False, z=1.6, center_dash=True):
    """管道：沿中心线的变宽度半透明带 + 端盖 + 虚线中心线。hw 可为标量或逐点数组。"""
    pts = np.asarray(pts, float)
    hw = np.broadcast_to(np.asarray(hw, float), (len(pts),))
    nrm = _normals(pts)
    left, right = pts + nrm * hw[:, None], pts - nrm * hw[:, None]
    poly = np.vstack([left, right[::-1]])
    ax.fill(poly[:, 0], poly[:, 1], color=PIPE, alpha=.24, lw=0, zorder=z)
    ax.plot(left[:, 0], left[:, 1], color=PIPE, lw=.9, zorder=z + .3)
    ax.plot(right[:, 0], right[:, 1], color=PIPE, lw=.9, zorder=z + .3)
    if center_dash:
        ax.plot(pts[:, 0], pts[:, 1], '--', color=DASH, lw=.8, zorder=z + .5)
    if cap_start:
        _endcap(ax, pts[0], -_normals(pts[:6])[0], hw[0], z)
    if cap_end:
        _endcap(ax, pts[-1], _normals(pts[-6:])[-1], hw[-1], z)


def pipe_side(ax, x0, x1, r=1.0, caps='lr', z=1.6):
    """侧视水平管道：矩形带 + 椭圆端盖 + 虚线轴线。"""
    ax.fill([x0, x1, x1, x0], [-r, -r, r, r], color=PIPE, alpha=.24, lw=0, zorder=z)
    ax.plot([x0, x1], [-r, -r], color=PIPE, lw=.9, zorder=z + .3)
    ax.plot([x0, x1], [r, r], color=PIPE, lw=.9, zorder=z + .3)
    for side in caps:
        x = x0 if side == 'l' else x1
        ax.add_patch(Ellipse((x, 0), .5, 2 * r, facecolor=CAP, edgecolor=PIPE, lw=.9, zorder=z + .4))
    ax.plot([x0, x1], [0, 0], '--', color=DASH, lw=.8, zorder=z + .5)


def pipe_section(ax, cx, cy, r, wall=.16, z=1.6):
    """端视/正视管道横截面：环形截面 + 十字中心线。"""
    ax.add_patch(Circle((cx, cy), r, facecolor=PIPE, alpha=.24, edgecolor='none', zorder=z))
    ax.add_patch(Circle((cx, cy), r - wall, facecolor='white', edgecolor='none', zorder=z + .1))
    ax.add_patch(Circle((cx, cy), r, facecolor='none', edgecolor=PIPE, lw=1.0, zorder=z + .2))
    for s in (-1, 1):
        ax.plot([cx - r * .55, cx + r * .55], [cy, cy], '--', color=DASH, lw=.7, zorder=z + .3)
        ax.plot([cx, cx], [cy - r * .55, cy + r * .55], '--', color=DASH, lw=.7, zorder=z + .3)


def capsule(ax, pts, hw, color=BLUE, alpha=1.0, z=4.2):
    """俯视机器人足迹：沿路径的圆头胶囊体，hw 可变宽（包绕半径随管道变化）。"""
    pts = np.asarray(pts, float)
    hw = np.broadcast_to(np.asarray(hw, float), (len(pts),))
    nrm = _normals(pts)
    left, right = pts + nrm * hw[:, None], pts - nrm * hw[:, None]
    poly = np.vstack([left, right[::-1]])
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, lw=0, zorder=z)
    for idx in (0, -1):
        ax.add_patch(Circle(tuple(pts[idx]), float(hw[idx]), facecolor=color, edgecolor='none', alpha=alpha, zorder=z))


def arch(ax, cx, r=1.38, a0=-38, a1=218, color=BLUE, lw=6.5, alpha=1.0, z=4.2):
    """侧视圆弧滚动体：包绕管道上方的弧段。"""
    t = np.radians(np.linspace(a0, a1, 160))
    ax.plot(cx + r * np.cos(t), r * np.sin(t), color=color, lw=lw, alpha=alpha,
            solid_capstyle='round', zorder=z)


def coil_side(ax, cx, loops=2, pitch=.78, r_out=1.32, color=BLUE, lw=4.6, alpha=1.0, z=4.2):
    """侧视缠绕螺旋：若干椭圆线圈。"""
    for i in range(loops):
        x = cx + (i - (loops - 1) / 2) * pitch
        ax.add_patch(Ellipse((x, 0), pitch * 1.02, 2 * r_out, facecolor='none',
                             edgecolor=color, lw=lw, alpha=alpha, zorder=z))


def head(ax, p0, p1, color=NOTE, lw=1.5, rad=0.0, z=8, scale=14, style='-|>'):
    ax.annotate('', xy=tuple(p1), xytext=tuple(p0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0,
                                mutation_scale=scale, connectionstyle=f'arc3,rad={rad}'), zorder=z)


def rot_arrow(ax, cx, cy, r, a0, a1, color=NOTE, lw=1.6, z=7):
    """绕圆心的自转箭头（a1>a0 为逆时针）。"""
    t = np.radians(np.linspace(a0, a1, 80))
    ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), color=color, lw=lw, zorder=z)
    tip = np.array([cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1))])
    tang = np.array([-math.sin(math.radians(a1)), math.cos(math.radians(a1))])
    head(ax, tip - .24 * tang, tip, color=color, lw=lw, z=z, scale=15)


def into_page(ax, cx, cy, s=.24, color=NOTE):
    """前进方向符号：⊙⊗ 中的 ⊗（指向纸内）。"""
    ax.add_patch(Circle((cx, cy), s, facecolor='white', edgecolor=color, lw=1.3, zorder=7))
    d = s * .707
    ax.plot([cx - d, cx + d], [cy - d, cy + d], color=color, lw=1.3, zorder=8)
    ax.plot([cx - d, cx + d], [cy + d, cy - d], color=color, lw=1.3, zorder=8)


def label(ax, text, xy, xytext, fontsize=10, color=NOTE, ha='left'):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fontsize, color=color, ha=ha, va='center',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.1, shrinkA=2, shrinkB=2), zorder=8)


def panel(ax, title, note=None, view=None):
    ax.set_title(title, fontsize=12.5, color=TITLE, pad=7)
    if view:
        ax.text(.025, .975, view, transform=ax.transAxes, fontsize=9.5, color=SUB, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#c9cfd2', lw=.7, alpha=.9), zorder=9)
    if note:
        ax.text(.5, -.05, note, transform=ax.transAxes, ha='center', va='top', fontsize=10.2, color=NOTE)
    ax.set_axis_off()
    ax.set_aspect('equal')


def save(fig, name):
    fig.savefig(HERE / f'{name}.png', dpi=160)
    fig.savefig(HERE / f'{name}.svg')
    plt.close(fig)
    print(name, 'ok')


# ---------------------------------------------------------------- 图 a：基本机理
def fig_a():
    fig = plt.figure(figsize=(13.6, 5.9), facecolor='white')
    fig.suptitle('圆弧滚动运动机理：包绕管道 → 绕管轴自转 → 沿管轴前进', fontsize=18, y=.988, color=TITLE)
    fig.text(.5, .922, '机理示意｜仅表达构型、接触使能与推进关系；不是仿真 / 实机结果，未含碰撞与动力学验证',
             ha='center', fontsize=10.5, color=SUB)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 2.05], left=.015, right=.985, top=.80, bottom=.20, wspace=.05)

    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-3.05, 3.05); ax.set_ylim(-2.75, 2.55)
    pipe_section(ax, 0, 0, 1.0)
    t = np.radians(np.linspace(-55, 215, 200))
    ax.plot(1.38 * np.cos(t), 1.38 * np.sin(t), color=BLUE, lw=7, solid_capstyle='round', zorder=4)
    td = np.radians(np.arange(-40, 201, 30))
    ax.scatter(1.38 * np.cos(td), 1.38 * np.sin(td), s=42, color=BLUE, edgecolor='white', linewidth=1.2, zorder=5)
    e = np.array([1.38 * math.cos(math.radians(215)), 1.38 * math.sin(math.radians(215))])
    tail = qbez(e, e + [-.55, -.35], e + [-.42, -1.05])
    ax.plot(tail[:, 0], tail[:, 1], color=PURPLE, lw=6, alpha=.4, solid_capstyle='round', zorder=4)
    rot_arrow(ax, 0, 0, 2.0, 20, 80)
    ax.text(.5, 2.28, '滚动自转', fontsize=10, color=NOTE, ha='center')
    into_page(ax, 2.0, -1.7)
    ax.text(2.0, -2.12, '前进方向（沿管轴）', fontsize=9, color=NOTE, ha='center', va='top')
    label(ax, '接触段（关节使能）', (-.89, 1.06), (-2.9, 2.0))
    label(ax, '非接触段（关节失能）', (-1.5, -1.55), (-2.9, -2.05))
    panel(ax, '① 端视：圆弧包绕管道，绕轴自转如轮',
          note='只有接触段关节使能，其余关节失能，形态始终保持圆弧', view='端视·沿管轴看')

    ax = fig.add_subplot(gs[1])
    ax.set_xlim(-6.6, 6.6); ax.set_ylim(-1.95, 3.2)
    pipe_side(ax, -6.25, 6.25, 1.0)
    for cx, al in [(-3.7, .28), (-1.0, .55), (1.7, 1.0)]:
        arch(ax, cx, alpha=al)
    head(ax, (1.05, 2.02), (2.5, 1.62), color=SUB, lw=1.3, rad=.35)
    ax.text(1.78, 2.35, '自转', fontsize=9.5, color=SUB, ha='center')
    head(ax, (-5.9, 2.72), (5.9, 2.72), color=NOTE, lw=1.7)
    ax.text(0, 2.92, '沿管轴向连续前进', fontsize=10.5, color=NOTE, ha='center')
    panel(ax, '② 侧视：自转一周前进一个导程',
          note='长直管上连续滚动推进，无障碍管段越长优势越明显', view='侧视')

    fig.text(.5, .105, '蓝＝圆弧包绕段（接触关节使能）　淡紫＝非接触段（关节失能）　灰＝管道', ha='center', fontsize=10.5)
    fig.text(.5, .042, '对应路线 a：长距离无障碍水平管上，相比螺旋爬行步态更节能、更高效', ha='center', fontsize=11, color=TITLE)
    save(fig, 'fig_a_rolling_basic')


# ---------------------------------------------------------------- 图 b：法兰盘跨越
def fig_b():
    fig = plt.figure(figsize=(16.6, 5.0), facecolor='white')
    fig.suptitle('法兰盘障碍跨越：圆弧滚动 ＋ 重力矩避障', fontsize=18, y=.988, color=TITLE)
    fig.text(.5, .918, '过程示意｜缠绕—解旋越障的阶段划分；不是仿真 / 实机结果，稳定性与通过率需实测验证',
             ha='center', fontsize=10.5, color=SUB)
    gs = fig.add_gridspec(1, 4, left=.012, right=.988, top=.79, bottom=.24, wspace=.045)

    tail_p = [(0.55, 1.38), (2.35, 2.95), (4.35, 0.95)]
    for k in range(4):
        ax = fig.add_subplot(gs[k])
        ax.set_xlim(-5.7, 5.7); ax.set_ylim(-2.4, 3.2)
        pipe_side(ax, -5.35, 5.35, 1.0)
        ax.fill([1.38, 1.82, 1.82, 1.38], [-1.6, -1.6, 1.6, 1.6], color='#b9c4c9', alpha=.55, lw=0, zorder=2.4)
        ax.plot([1.38, 1.82], [1.6, 1.6], color=PIPE, lw=1, zorder=2.5)
        ax.plot([1.38, 1.82], [-1.6, -1.6], color=PIPE, lw=1, zorder=2.5)
        if k == 0:
            arch(ax, -3.5)
            head(ax, (-4.9, 1.95), (-2.4, 1.95), color=NOTE, lw=1.6)
            panel(ax, '① 滚动接近', note='圆弧滚动接近法兰盘障碍', view='侧视')
        elif k == 1:
            arch(ax, -3.5, alpha=.15)
            coil_side(ax, -0.7)
            label(ax, '缠绕成螺旋（重力矩避障形态）', (-0.7, 1.42), (0.3, 2.55), ha='center')
            panel(ax, '② 缠绕成螺旋', note='停止滚动，缠绕成重力矩避障所需螺旋', view='侧视')
        elif k == 2:
            coil_side(ax, 0.55)
            tail = qbez(*tail_p)
            ax.plot(tail[:, 0], tail[:, 1], color=PURPLE, lw=5.5, solid_capstyle='round', zorder=4.2)
            head(ax, tail[-8], tail[-1], color=PURPLE, lw=2.2, scale=16)
            label(ax, '尾部抬起越过法兰', (2.55, 2.12), (3.7, 2.8), ha='right')
            label(ax, '重力矩解旋', (-0.02, -1.32), (-1.65, -2.05))
            panel(ax, '③ 解旋越障', note='重力矩解旋：尾部抬起越过法兰盘', view='侧视')
        else:
            coil_side(ax, 0.55, alpha=.15)
            tail = qbez(*tail_p)
            ax.plot(tail[:, 0], tail[:, 1], color=PURPLE, lw=5.5, alpha=.15, solid_capstyle='round', zorder=4.2)
            arch(ax, 3.6)
            head(ax, (2.1, 1.95), (4.7, 1.95), color=NOTE, lw=1.6)
            panel(ax, '④ 回归滚动', note='越障完成：恢复圆弧滚动高效推进', view='侧视')

    fig.text(.5, .13, '蓝＝圆弧滚动 / 螺旋形态（接触使能）　紫＝缠绕 / 解旋的后身（失能随动）　灰＝管道与法兰盘', ha='center', fontsize=10.5)
    fig.text(.5, .055, '对应路线 b：非连续水平直管的法兰盘障碍，高效率、稳定低成本跨越', ha='center', fontsize=11, color=TITLE)
    save(fig, 'fig_b_flange_crossing')


# ---------------------------------------------------------------- 图 c：弯管跨越（俯视）
def fig_c():
    fig = plt.figure(figsize=(14.6, 5.6), facecolor='white')
    fig.suptitle('弯管道跨越：非连续弯管跨隙 ＆ 连续柔弯顺行', fontsize=18, y=.988, color=TITLE)
    fig.text(.5, .918, '俯视过程示意｜表达圆弧滚动对弯道的通过方式；不是仿真 / 实机结果',
             ha='center', fontsize=10.5, color=SUB)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1.28], left=.015, right=.985, top=.79, bottom=.21, wspace=.04)

    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-6.3, 5.4); ax.set_ylim(-2.15, 6.15)
    curved_band(ax, np.linspace([-5.7, 0], [-1.2, 0], 30), 1.0, cap_start=True)
    curved_band(ax, np.linspace([0.5, 0.85], [3.84, 4.83], 30), 1.0, cap_end=True)
    bridge = qbez((-0.9, 0), (0.35, 0.55), (1.35, 1.6))
    path = np.vstack([np.linspace([-4.85, 0], [-0.9, 0], 40), bridge[1:]])
    capsule(ax, path, 0.62)
    head(ax, (1.0, 1.22), (1.72, 1.85), color=BLUE, lw=2.4, scale=16)
    label(ax, '跨越管段间隙', (0.1, 0.38), (1.9, -1.25))
    label(ax, '全程保持圆弧滚动形态', (-3.3, 0.66), (-5.85, 2.0))
    panel(ax, '① 非连续弯管道（有间隙）',
          note='圆弧滚动直接跨越管段间隙，随后进入下一段继续滚动', view='俯视')

    ax = fig.add_subplot(gs[1])
    ax.set_xlim(-6.5, 3.9); ax.set_ylim(-1.65, 5.65)
    seg1 = np.linspace([-5.9, 0], [-1.5, 0], 40)
    arc = arc_pts((-1.5, 3.4), 3.4, -90, -28)
    u2 = np.array([math.cos(math.radians(62)), math.sin(math.radians(62))])
    seg2 = arc[-1] + np.linspace(0, 1, 30)[:, None] * u2 * 2.4
    pipe = np.vstack([seg1, arc[1:], seg2[1:]])
    curved_band(ax, pipe, 1.0, cap_start=True, cap_end=True)
    robot = pipe[9:147]
    n = len(robot)
    capsule(ax, robot[:n // 3 + 1], 0.62, alpha=.28)
    capsule(ax, robot[n // 3:2 * n // 3 + 1], 0.62, alpha=.55)
    capsule(ax, robot[2 * n // 3:], 0.62)
    tang = robot[-1] - robot[-4]
    tang = tang / np.linalg.norm(tang)
    head(ax, robot[-4], robot[-1] + tang * .55, color=BLUE, lw=2.4, scale=16)
    label(ax, '接触关节随弯自适应', (-.5, 0.2), (-3.8, 3.4), ha='center')
    panel(ax, '② 连续柔和弯管道',
          note='无需改变步态，滚动顺弯连续通过', view='俯视')

    fig.text(.5, .115, '蓝＝机器人（包绕管顶、接触使能）　淡色＝先前的滚动位置　灰＝管道', ha='center', fontsize=10.5)
    fig.text(.5, .045, '对应路线 c：非连续弯管道与连续柔和弯管道上，高效率、更节能跨过障碍', ha='center', fontsize=11, color=TITLE)
    save(fig, 'fig_c_elbow_crossing')


# ---------------------------------------------------------------- 图 d：变径弯管与局部顺应性
def fig_d():
    fig = plt.figure(figsize=(15.2, 5.6), facecolor='white')
    fig.suptitle('变径弯管跨越：圆弧滚动 ＋ 局部顺应性控制（滑动窗口）', fontsize=18, y=.988, color=TITLE)
    fig.text(.5, .918, '机理示意｜表达包绕半径随管道放大、使能关节随接触增多；不是仿真 / 实机结果',
             ha='center', fontsize=10.5, color=SUB)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.32, 1.0], left=.015, right=.985, top=.79, bottom=.21, wspace=.05)

    # 端视 · 小径段
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.6, 2.6)
    pipe_section(ax, 0, 0, 0.85)
    t = np.radians(np.linspace(-50, 215, 200))
    ax.plot(1.15 * np.cos(t), 1.15 * np.sin(t), color=BLUE, lw=6.5, solid_capstyle='round', zorder=4)
    td = np.radians(np.arange(-35, 201, 30))
    ax.scatter(1.15 * np.cos(td), 1.15 * np.sin(td), s=38, color=BLUE, edgecolor='white', linewidth=1.1, zorder=5)
    rot_arrow(ax, 0, 0, 1.66, 15, 85)
    label(ax, '包绕半径 R1', (1.15 * math.cos(math.radians(150)), 1.15 * math.sin(math.radians(150))), (-2.65, 1.9))
    panel(ax, '① 小径段：包绕半径小', note='端视：圆弧贴合小径管道，使能关节较少', view='端视')

    # 俯视 · 渐扩弯管 + 局部使能窗口
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(-5.4, 4.6); ax.set_ylim(-2.05, 5.05)
    seg1 = np.linspace([-4.75, 0], [-1.2, 0], 40)
    arc = arc_pts((-1.2, 3.2), 3.2, -90, -30)
    u2 = np.array([math.cos(math.radians(60)), math.sin(math.radians(60))])
    seg2 = arc[-1] + np.linspace(0, 1, 30)[:, None] * u2 * 2.6
    pipe = np.vstack([seg1, arc[1:], seg2[1:]])
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pipe, axis=0).T))])
    hw_pipe = 0.85 + (1.35 - 0.85) * s / s[-1]
    curved_band(ax, pipe, hw_pipe, cap_start=True, cap_end=True)
    robot = np.vstack([seg1[5:], arc[1:], arc[-1] + np.linspace(0, 1, 20)[:, None] * u2 * 0.6])
    sr = 0.4 + np.concatenate([[0], np.cumsum(np.hypot(*np.diff(robot, axis=0).T))])
    hw_rob = 0.85 + (1.35 - 0.85) * sr / s[-1] - 0.4
    capsule(ax, robot[:17], hw_rob[:17], color=PURPLE, alpha=.45)
    capsule(ax, robot[16:143], hw_rob[16:143], color=BLUE)
    capsule(ax, robot[142:], hw_rob[142:], color=ORANGE)
    head(ax, robot[-5], robot[-1] + u2 * .45, color=ORANGE, lw=2.4, scale=16)
    ax.plot([-2.8, -2.8], [3.0, 3.28], color=NOTE, lw=1.1, zorder=8)
    ax.plot([-2.8, 1.2], [3.28, 3.28], color=NOTE, lw=1.1, zorder=8)
    ax.plot([1.2, 1.2], [3.28, 3.0], color=NOTE, lw=1.1, zorder=8)
    ax.text(-0.8, 3.58, '局部使能窗口（随接触增大而扩大）', fontsize=10, color=TITLE, ha='center')
    label(ax, '失能段（保持包绕）', (-3.6, 0.55), (-3.7, 1.98), ha='center')
    label(ax, '过渡扩展', (1.85, 2.0), (3.4, 0.8), ha='center')
    head(ax, (-4.6, -0.85), (-4.6, 0.85), color=SUB, lw=1.2, style='<|-|>', scale=12)
    ax.text(-4.6, -1.32, '小径端', fontsize=9.5, color=SUB, ha='center')
    n_end = np.array([-math.sin(math.radians(60)), math.cos(math.radians(60))])
    c_end = arc[-1] + u2 * 0.9
    head(ax, c_end - 1.33 * n_end, c_end + 1.33 * n_end, color=SUB, lw=1.2, style='<|-|>', scale=12)
    ax.text(3.55, 1.3, '大径端', fontsize=9.5, color=SUB, ha='center')
    panel(ax, '② 渐扩弯管：局部顺应性窗口',
          note='俯视：管径渐扩，使能窗口随接触关节增多而扩大', view='俯视')

    # 端视 · 大径段
    ax = fig.add_subplot(gs[2])
    ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.6, 2.6)
    pipe_section(ax, 0, 0, 1.35)
    t = np.radians(np.linspace(-55, 220, 200))
    ax.plot(1.75 * np.cos(t), 1.75 * np.sin(t), color=BLUE, lw=6.5, solid_capstyle='round', zorder=4)
    td = np.radians(np.arange(-40, 206, 24))
    ax.scatter(1.75 * np.cos(td), 1.75 * np.sin(td), s=38, color=BLUE, edgecolor='white', linewidth=1.1, zorder=5)
    rot_arrow(ax, 0, 0, 2.32, 15, 80)
    label(ax, '包绕半径放大 R2', (1.75 * math.cos(math.radians(140)), 1.75 * math.sin(math.radians(140))), (-2.7, 2.05))
    panel(ax, '③ 大径段：半径放大、关节增多', note='端视：弧随管道放大，接触关节即时增多', view='端视')

    fig.text(.5, .115, '蓝＝局部使能窗口（顺应控制）　橙＝正在扩展的过渡关节　淡紫＝失能段（保持包绕）　灰＝渐扩管道', ha='center', fontsize=10.5)
    fig.text(.5, .045, '对应路线 d：异径弯管上圆弧不掉管、随管径放大，实现变径＋弯道跨越', ha='center', fontsize=11, color=TITLE)
    save(fig, 'fig_d_variable_radius')


# ---------------------------------------------------------------- 图 e：T 型管跨越与分支选择
def fig_e():
    fig = plt.figure(figsize=(16.4, 5.8), facecolor='white')
    fig.suptitle('T 型管道：跨管避障与分支选择（圆弧滚动 ＋ 重力矩避障）', fontsize=18, y=.988, color=TITLE)
    fig.text(.5, .918, '过程示意｜跨越与选支的阶段划分；不是仿真 / 实机结果',
             ha='center', fontsize=10.5, color=SUB)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0], left=.013, right=.987, top=.79, bottom=.22, wspace=.05)

    # 侧视 · 向下凸出
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-5.9, 5.9); ax.set_ylim(-5.3, 3.0)
    curved_band(ax, np.linspace([0, -1.0], [0, -4.6], 20), 1.0, cap_end=True)
    pipe_side(ax, -5.5, 5.5, 1.0)
    for cx, al in [(-3.7, .28), (-1.2, .55), (1.3, 1.0)]:
        arch(ax, cx, alpha=al)
    head(ax, (-4.9, 2.0), (3.9, 2.0), color=NOTE, lw=1.6)
    label(ax, '直接滚过', (1.95, 1.1), (3.6, 2.55))
    label(ax, '向下凸出分支', (0.85, -2.6), (3.0, -3.9))
    panel(ax, '① 向下凸出 T 型管：直接滚过',
          note='侧视：保持圆弧滚动形态直接滚过向下分支，无需变形', view='侧视')

    # 正视 · 两侧凸出：抬尾越障
    tailp = [(1.0, -1.0), (2.8, 3.6), (4.9, -0.55)]
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(-5.7, 5.7); ax.set_ylim(-3.5, 3.5)
    curved_band(ax, np.linspace([-1, 0], [-5.2, 0], 20), 1.0, cap_end=True)
    curved_band(ax, np.linspace([1, 0], [5.2, 0], 20), 1.0, cap_end=True)
    pipe_section(ax, 0, 0, 1.0)
    tt = np.linspace(0, 2 * math.pi, 200)
    ax.plot(1.42 * np.cos(tt), 1.42 * np.sin(tt), color=BLUE, lw=6, zorder=4)
    tail = qbez(*tailp)
    ax.plot(tail[:, 0], tail[:, 1], color=PURPLE, lw=5.5, solid_capstyle='round', zorder=4.2)
    head(ax, tail[-8], tail[-1], color=PURPLE, lw=2.2, scale=16)
    label(ax, '单圈形态（重力矩构型）', (-1.0, 1.0), (-4.4, 2.55))
    label(ax, '尾部抬起越过凸起侧管', (2.87, 1.5), (2.6, 2.85), ha='center')
    panel(ax, '② 两侧凸出 T 型管：抬尾越障',
          note='正视：单圈解旋，尾部抬起越过凸起侧管道', view='正视·沿主管轴')

    # 正视 · 两侧凸出：分支选择
    ax = fig.add_subplot(gs[2])
    ax.set_xlim(-5.7, 5.7); ax.set_ylim(-3.5, 3.5)
    curved_band(ax, np.linspace([-1, 0], [-5.2, 0], 20), 1.0, cap_end=True)
    curved_band(ax, np.linspace([1, 0], [5.2, 0], 20), 1.0, cap_end=True)
    pipe_section(ax, 0, 0, 1.0)
    tgap = np.radians(np.linspace(200, 520, 220))
    ax.plot(1.42 * np.cos(tgap), 1.42 * np.sin(tgap), color=BLUE, lw=5.5, alpha=.38, zorder=4)
    body = np.linspace([-0.9, 0], [-4.55, 0], 40)
    capsule(ax, body, 0.55)
    head(ax, body[-3], [-4.85, 0], color=BLUE, lw=2.4, scale=16)
    label(ax, '解旋', (1.0, -1.0), (2.4, -2.5))
    label(ax, '身体送入目标分支（无需抬尾）', (-2.6, 0.15), (-4.5, 2.55))
    label(ax, '另一侧分支保留', (3.0, 0.95), (3.6, 2.3))
    panel(ax, '③ 两侧凸出 T 型管：分支选择',
          note='正视：单圈解旋将身体送入目标分支，完成分支选择', view='正视·沿主管轴')

    fig.text(.5, .125, '蓝＝圆弧滚动 / 单圈形态（使能）　紫＝解旋随动的尾部（失能）　灰＝管道与分支', ha='center', fontsize=10.5)
    fig.text(.5, .05, '对应路线 e：向下凸出直接滚过；两侧凸出经单圈形态抬尾越障或解旋选支，最终回归圆弧滚动', ha='center', fontsize=11, color=TITLE)
    save(fig, 'fig_e_tee_crossing')


if __name__ == '__main__':
    fig_a()
    fig_b()
    fig_c()
    fig_d()
    fig_e()
