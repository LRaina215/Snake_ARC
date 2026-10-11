"""Rolling Helix-Arc 路线 §2：可重复绘制的三维研究设想图。

以路线中的搭管照片为构型依据，参考 Snake_Robot_Gait/images/
Climbing_over_a_Flange_CaseB.png 的实体曲管、分段配色与几何注释方式。
所有管道和机器人均由三维曲面生成，统一正交侧视投影、全场景深度排序、
曲面光照；PNG 与 SVG 来自同一几何。没有运行仿真或计算接触力。
机器人是连续骨干曲线的分节示意，不是指定硬件的逆运动学解。
"""
from pathlib import Path
import argparse
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
FONT = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': FONT.get_name(), 'axes.unicode_minus': False,
                     'svg.fonttype': 'path', 'savefig.facecolor': 'white'})
BLUE = '#246faa'
PURPLE = '#79639f'
ORANGE = '#e7a039'
PIPE = '#c4ced5'
INK = '#223d50'
MUTED = '#617786'
JOINT = '#465765'
GREEN = '#428d79'
BODY_R = .092
BODY_LENGTH = 5.20


def unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def bezier(points, n=50):
    p = np.asarray(points, float)
    t = np.linspace(0, 1, n)
    m = len(p) - 1
    return sum(math.comb(m, k) * ((1-t)**(m-k)*t**k)[:, None] * p[k]
               for k in range(m+1))


def length(p):
    return np.linalg.norm(np.diff(p, axis=0), axis=1).sum()


def sample(p, s):
    p = np.asarray(p, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return np.column_stack([np.interp(s, d, p[:, k]) for k in range(3)])


def straight(p0, p1, n=25):
    return np.linspace(p0, p1, n)


class Scene:
    def __init__(self, camera=(3, -10, 5)):
        self.eye = unit(camera)
        self.right = unit(np.cross([0, 0, 1], self.eye))
        self.up = unit(np.cross(self.eye, self.right))
        self.basis = np.stack([self.right, self.up, self.eye], axis=1)
        self.faces = []
        self.colors = []
        self.light = unit([-4, -7, 10])

    def project(self, p):
        return np.asarray(p) @ self.basis

    def mesh(self, vertices, faces, color, light=True):
        v = np.asarray(vertices, float)
        f = np.asarray(faces, int)
        polys = v[f]
        normal = unit(np.cross(polys[:, 1]-polys[:, 0], polys[:, 2]-polys[:, 0]))
        base = np.asarray(to_rgb(color))
        diffuse = np.clip(normal @ self.light, 0, 1)
        # Moderate fill light preserves readable state colors on the far side.
        shade = .57 + .40 * diffuse if light else np.ones(len(f))
        rgb = np.minimum(1, base[None, :] * shade[:, None])
        halfway = unit(self.light + self.eye)
        shine = np.clip(normal @ halfway, 0, 1)**24 * .15
        rgb = rgb + shine[:, None] * (1-rgb)
        self.faces.extend(self.project(polys))
        self.colors.extend(rgb)

    def sweep(self, p, r, color, sectors=40, hollow=False):
        p = np.asarray(p, float)
        tangents = unit(np.gradient(p, axis=0))
        # All pipe axes lie in the horizontal plane except explicit T downspouts.
        # Project a fixed reference to obtain stable frames on the planar curves.
        ref = np.array([0., 0., 1.])
        if abs(tangents[:, 2]).max() > .94:
            ref = np.array([0., 1., 0.])
        n = unit(np.cross(ref, tangents))
        b = unit(np.cross(tangents, n))
        radius = np.broadcast_to(np.asarray(r, float), (len(p),))
        a = np.arange(sectors)*2*np.pi/sectors
        directions = n[:, None, :]*np.cos(a)[None, :, None] + b[:, None, :]*np.sin(a)[None, :, None]
        rings = p[:, None, :] + radius[:, None, None]*directions
        verts = rings.reshape(-1, 3)
        faces = []
        for i in range(len(p)-1):
            for j in range(sectors):
                k = (j+1) % sectors
                faces.append([i*sectors+j, i*sectors+k, (i+1)*sectors+k, (i+1)*sectors+j])
        self.mesh(verts, faces, color)
        if hollow:
            inner = p[:, None, :] + (radius*.83)[:, None, None]*directions
            inner_faces = [f[::-1] for f in faces]
            self.mesh(inner.reshape(-1, 3), inner_faces, '#889baa')
            # Annular end faces visibly distinguish an open pipe from a solid rod.
            for i, sign in [(0, -1), (len(p)-1, 1)]:
                v = np.vstack([rings[i], inner[i]])
                f = [[j, (j+1)%sectors, sectors+(j+1)%sectors, sectors+j] for j in range(sectors)]
                if sign > 0:
                    f = [row[::-1] for row in f]
                self.mesh(v, f, '#d4dde3')
        else:
            for i, sign in [(0, -1), (len(p)-1, 1)]:
                v = np.vstack([rings[i], p[i]])
                f = [[sectors, j, (j+1)%sectors] for j in range(sectors)]
                if sign < 0:
                    f = [row[::-1] for row in f]
                self.mesh(v, f, color)

    def ball(self, center, r, color, sectors=16):
        a = np.arange(sectors)*2*np.pi/sectors
        z = np.linspace(-np.pi/2+.02, np.pi/2-.02, 9)
        v = np.array([[np.cos(t)*np.cos(q), np.cos(t)*np.sin(q), np.sin(t)]
                      for t in z for q in a])*r + center
        f = []
        for i in range(len(z)-1):
            for j in range(sectors):
                k = (j+1)%sectors
                f.append([i*sectors+j, i*sectors+k, (i+1)*sectors+k, (i+1)*sectors+j])
        self.mesh(v, f, color)

    def pipe(self, p0, p1=None, radius=.55):
        p = p0 if p1 is None else straight(p0, p1)
        self.sweep(p, radius, PIPE, sectors=48, hollow=True)

    def sharp_elbow(self, radius=.43):
        """One continuous hollow pipe with an exact 90-degree mitered bend.

        The enlarged diagonal ring joins the two cylinder walls without a gap
        or an internal end cap. This is an idealized sharp bend, not a smooth elbow.
        """
        sectors = 48
        angles = np.arange(sectors)*2*np.pi/sectors
        u, v = np.cos(angles), np.sin(angles)
        centers = np.vstack([straight([-2.5, 0, 0], [-.55, 0, 0], 20),
                             [[0, 0, 0]], straight([0, .55, 0], [0, 2.60, 0], 20)])
        def rings(r):
            incoming = np.array([np.column_stack([np.full(sectors, x), r*u, r*v])
                                 for x in centers[:20, 0]])
            corner = np.column_stack([-r*u, r*u, r*v])[None, :, :]
            outgoing = np.array([np.column_stack([-r*u, np.full(sectors, y), r*v])
                                 for y in centers[21:, 1]])
            return np.vstack([incoming, corner, outgoing])
        outer, inner = rings(radius), rings(radius*.83)
        faces = []
        for i in range(len(outer)-1):
            for j in range(sectors):
                k = (j+1)%sectors
                faces.append([i*sectors+j, i*sectors+k, (i+1)*sectors+k, (i+1)*sectors+j])
        self.mesh(outer.reshape(-1, 3), faces, PIPE)
        self.mesh(inner.reshape(-1, 3), [f[::-1] for f in faces], '#889baa')
        for i in (0, len(outer)-1):
            vertices = np.vstack([outer[i], inner[i]])
            cap = [[j, (j+1)%sectors, sectors+(j+1)%sectors, sectors+j]
                   for j in range(sectors)]
            if i != 0:
                cap = [f[::-1] for f in cap]
            self.mesh(vertices, cap, '#d4dde3')

    def robot(self, p, state, modules=26, pale=False):
        """state maps material arclength to the intended control state."""
        p = np.asarray(p, float)
        d = length(p)
        knots = np.linspace(0, d, modules+1)
        for a, b in zip(knots[:-1], knots[1:]):
            color = state((a+b)/2)
            if pale:
                color = np.asarray(to_rgb(color))*.28 + .72
            q = sample(p, np.linspace(a+.009, b-.009, 6))
            self.sweep(q, BODY_R, color, sectors=16)
        for s in knots:
            self.ball(sample(p, [s])[0], BODY_R*.89, '#bcc8d0' if pale else JOINT)

    def draw(self, ax, bounds):
        # Sort every pipe and robot surface together so the near/far wraps occlude
        # correctly. Independent matplotlib 3-D artists cannot provide this order.
        depth = np.array([row[:, 2].mean() for row in self.faces])
        order = np.argsort(depth, kind='stable')
        polygons = [self.faces[i][:, :2] for i in order]
        colors = np.asarray(self.colors)[order]
        collection = PolyCollection(polygons, facecolors=colors, edgecolors=colors,
                                    linewidths=.025, antialiased=False, closed=True)
        ax.add_collection(collection)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal')
        ax.axis('off')


def saddle(center, tangent=(1, 0, 0), radius=.55, total=BODY_LENGTH,
           active_color=BLUE, transition=False, tilt=.23):
    """Top semicircle + two gravity-hanging ends, consistent with the source photo.

    Fixed total backbone length: as pipe radius grows, arc consumes more material
    and passive ends become shorter. No numerical contact or enable rule is inferred.
    """
    c = np.asarray(center, float)
    t = unit(tangent)
    n = unit(np.cross([0, 0, 1], t))
    z = np.array([0., 0., 1.])
    r = radius + BODY_R + .010
    theta = np.linspace(np.pi, 0, 110)
    core = c + r*np.cos(theta)[:, None]*n + r*np.sin(theta)[:, None]*z
    core += ((theta/np.pi-.5)*tilt)[:, None]*t
    remaining = max(.30, total-length(core))
    tail = remaining/2
    h = np.linspace(0, 1, 40)
    first = core[0] - (1-h)[:, None]*tail*z - (.055*np.sin(np.pi*h))[:, None]*n
    last = core[-1] - h[:, None]*tail*z + (.055*np.sin(np.pi*h))[:, None]*n
    p = np.vstack([first, core[1:], last[1:]])
    a = length(first)
    b = a + length(core)
    def state(s):
        if a <= s <= b:
            if transition and (s-a < .22 or b-s < .22):
                return ORANGE
            return active_color
        return PURPLE
    return p, state, core


def coil(center_x=-.85, radius=.55):
    r = radius+BODY_R+.010
    theta = np.linspace(np.pi, 3*np.pi, 180)
    core = np.column_stack([center_x+np.linspace(-.40, .40, len(theta)),
                            r*np.cos(theta), r*np.sin(theta)])
    remain = max(.3, BODY_LENGTH-length(core))
    first = straight(core[0]-[0, 0, remain/2], core[0], 25)
    last = straight(core[-1], core[-1]-[0, 0, remain/2], 25)
    p = np.vstack([first, core[1:], last[1:]])
    lo, hi = length(first), length(first)+length(core)
    return p, lambda s: BLUE if lo < s < hi else ORANGE


def lifted_transfer(center_x=.95):
    """Front arc provides target-side support; a raised rear segment clears the rim."""
    _, _, arc = saddle([center_x, 0, 0], total=BODY_LENGTH)
    start = np.array([-1.65, -.66, -1.15])
    lift = bezier([start, [-1.55, -.70, 1.35], [-.20, -.76, 1.88],
                   [.50, -.76, 1.86], arc[0]], 100)
    remain = max(.22, BODY_LENGTH-length(lift)-length(arc))
    tail = straight(arc[-1], arc[-1]-[0, 0, remain], 25)
    p = np.vstack([lift, arc[1:], tail[1:]])
    lo, hi = length(lift), length(lift)+length(arc)
    return p, lambda s: BLUE if lo < s < hi else (ORANGE if s < lo else PURPLE), lift


def arrow(ax, scene, p0, p1, color=INK, rad=0, style='-|>', lw=1.65):
    a, b = scene.project([p0, p1])[:, :2]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=15,
                                color=color, linewidth=lw, shrinkA=0, shrinkB=0,
                                connectionstyle=f'arc3,rad={rad}', zorder=20))


def callout(ax, scene, text, target, xy, color=INK, align='left'):
    p = scene.project(target)[:2]
    ax.annotate(text, xy=p, xytext=xy, textcoords='axes fraction',
                fontsize=11.0, color=color, ha=align, va='center',
                arrowprops={'arrowstyle': '-', 'color': color, 'lw': 1.15,
                            'shrinkA': 3, 'shrinkB': 3,
                            'connectionstyle': 'angle3,angleA=0,angleB=90'},
                bbox={'boxstyle': 'round,pad=.26', 'fc': 'white', 'ec': 'none', 'alpha': .96},
                zorder=25)


def text_at(ax, scene, text, p, color=MUTED, size=10, **kwargs):
    xy = scene.project(p)[:2]
    ax.text(*xy, text, color=color, fontsize=size, ha='center', va='center',
            zorder=25, bbox={'fc': 'white', 'ec': 'none', 'alpha': .86, 'pad': 1.4}, **kwargs)


def rotation(ax, scene, x, r=.99):
    a = np.linspace(.18*np.pi, .74*np.pi, 45)
    p = np.column_stack([np.full(len(a), x), r*np.cos(a), r*np.sin(a)])
    q = scene.project(p)
    ax.plot(q[:, 0], q[:, 1], color=INK, lw=1.5, zorder=20)
    arrow(ax, scene, p[-6], p[-1], lw=1.5)


def figure(title, subtitle, rows, cols, notes, figsize=None):
    if figsize is None:
        figsize = (15.6, 8.0 if rows == 1 else 13.2)
    fig = plt.figure(figsize=figsize)
    fig.text(.045, .966, title, fontsize=20, color=INK, va='top', weight='bold')
    fig.text(.045, .918 if rows == 1 else .934, subtitle, fontsize=11, color=MUTED, va='top')
    top = .855 if rows == 1 else .880
    bottom = .180 if rows == 1 else .134
    left, right, gapx, gapy = .024, .978, .026, .043
    w = (right-left-(cols-1)*gapx)/cols
    h = (top-bottom-(rows-1)*gapy)/rows
    axes = []
    for i, (heading, note) in enumerate(notes):
        row, col = divmod(i, cols)
        x = left+col*(w+gapx)
        y = top-(row+1)*h-row*gapy
        fig.patches.append(Rectangle((x, y), w, h, transform=fig.transFigure,
                                    fc='#fafcfd', ec='#e3eaee', lw=.8, zorder=-10))
        fig.text(x+.018, y+h-.022, heading, fontsize=13.2, color=INK,
                 weight='bold', va='top')
        fig.text(x+w/2, y+.018, note, fontsize=10.3, color=MUTED,
                 ha='center', va='bottom', linespacing=1.45)
        if rows == 1:
            ax = fig.add_axes([x+.009, y+.077, w-.018, h-.139])
        else:
            ax = fig.add_axes([x+.009, y+.047, w-.018, h-.103])
        axes.append(ax)
    legend_y = .099 if rows == 1 else .073
    for x, c, label in [(.055, BLUE, '拟使能的接触段'), (.322, PURPLE, '失能悬垂段'),
                         (.576, ORANGE, '变形 / 顺应调整段'), (.831, PIPE, '管道')]:
        fig.patches.append(Rectangle((x, legend_y), .018, .015, transform=fig.transFigure,
                                    fc=c, ec='none'))
        fig.text(x+.025, legend_y+.008, label, fontsize=10.4, color=INK, va='center')
    fig.text(.045, .035, '研究设想的几何与阶段示意；稳定性、碰撞裕度和效率仍需验证。',
             fontsize=10, color=MUTED)
    return fig, axes


def save(fig, name):
    fig.savefig(HERE/(name+'.png'), dpi=180)
    fig.savefig(HERE/(name+'.svg'))
    plt.close(fig)
    print(name, flush=True)


def fig_a():
    fig, axs = figure('a  长距离水平管：圆弧搭管与局部使能',
                       '三维侧视｜顶部圆弧接触管道，两侧非接触段在重力下悬垂；沿管轴滚动推进为研究目标。',
                       1, 2, [('① 搭管构型与控制分区', '用接触段建立包绕；非接触关节失能、随动'),
                              ('② 长直管上的推进设想', '同一机器人在两个时刻的示意位置；轴向推进关系待验证')])
    s = Scene((4, -10, 4.4))
    s.pipe([-2.25, 0, 0], [2.25, 0, 0])
    p, state, core = saddle([0, 0, 0])
    s.robot(p, state)
    s.draw(axs[0], (-2.8, 2.8, -2.35, 1.60))
    callout(axs[0], s, '顶部圆弧\n接触关节使能', core[53], (.04, .89), BLUE)
    callout(axs[0], s, '两侧悬垂段\n非接触关节失能', p[17], (.72, .18), PURPLE)
    callout(axs[0], s, '分节连杆与关节', p[28], (.03, .18))
    rotation(axs[0], s, .04, .98)
    text_at(axs[0], s, '滚动相位', [.05, .07, 1.13], size=10.0)
    arrow(axs[0], s, [1.80, 0, -.28], [1.80, 0, -.81], color=MUTED)
    text_at(axs[0], s, 'g', [2.00, 0, -.55], size=12)

    s = Scene((4, -10, 4.4))
    s.pipe([-3.1, 0, 0], [3.1, 0, 0])
    p, state, _ = saddle([-1.50, 0, 0])
    s.robot(p, state, pale=True)
    p, state, _ = saddle([1.40, 0, 0])
    s.robot(p, state)
    s.draw(axs[1], (-3.45, 3.45, -2.35, 1.60))
    arrow(axs[1], s, [-1.50, 0, 1.20], [1.40, 0, 1.20], color=BLUE)
    text_at(axs[1], s, '沿管轴前进', [0, 0, 1.40], color=BLUE, size=11)
    text_at(axs[1], s, '$t_0$', [-1.70, -.65, -1.9], size=12)
    text_at(axs[1], s, '$t_1$', [1.40, -.65, -1.9], color=BLUE, size=12)
    save(fig, 'fig_a_rolling_basic_3d')


def flange_scene():
    s = Scene()
    s.pipe([-2.65, 0, 0], [2.65, 0, 0])
    s.sweep(straight([-.14, 0, 0], [.14, 0, 0], 2), .94, '#9aabb8', sectors=64)
    # Bolt heads lie on the near axial face and reinforce the flange's disk shape.
    for t in np.linspace(0, 2*np.pi, 8, endpoint=False):
        c = [.155, .78*np.cos(t), .78*np.sin(t)]
        s.sweep(straight(c, np.array(c)+[.045, 0, 0], 2), .055, JOINT, sectors=6)
    return s


def fig_b():
    fig, axs = figure('b  法兰盘：圆弧滚动与重力矩避障的衔接',
                       '统一三维侧视｜盘缘是明确的空间障碍；抬尾段单独标出，控制方式需进一步设计。',
                       2, 2, [('① 滚动接近法兰', '顶部接触段支撑，两侧悬垂；盘缘限制直接通过'),
                              ('② 停滚，转为螺旋包绕', '形成重力矩避障所需构型，准备解旋'),
                              ('③ 解旋与抬尾越过盘缘', '示意前部建立目标侧支撑，后部抬起并跨过法兰'),
                              ('④ 恢复圆弧滚动', '跨越后重建顶部圆弧与两侧悬垂构型')])
    for k, ax in enumerate(axs):
        s = flange_scene()
        if k == 0:
            p, st, _ = saddle([-1.7, 0, 0])
        elif k == 1:
            p, st = coil(-.85)
        elif k == 2:
            p, st, lift = lifted_transfer(.95)
        else:
            p, st, _ = saddle([1.65, 0, 0])
        s.robot(p, st)
        s.draw(ax, (-3.10, 3.10, -2.20, 1.58))
        if k == 0:
            arrow(ax, s, [-2.3, 0, 1.2], [-1.1, 0, 1.2], color=BLUE)
            callout(ax, s, '法兰盘 / 盘缘', [0, -.5, .78], (.76, .72))
        elif k == 1:
            callout(ax, s, '螺旋包绕段', [-.85, -.2, .59], (.02, .87), BLUE)
            arrow(ax, s, [-1.18, -.72, -.30], [-.70, -.80, .10], rad=-.35, color=ORANGE)
            text_at(ax, s, '解旋准备', [-1.15, -1.0, -1.05], color=ORANGE)
        elif k == 2:
            callout(ax, s, '尾部抬起越过盘缘', lift[53], (.07, .93), ORANGE)
            callout(ax, s, '目标侧接触支撑', [1.05, 0, .65], (.68, .23), BLUE)
            arrow(ax, s, [-.60, -.77, 1.38], [.34, -.77, 1.38], color=ORANGE, rad=-.15)
        else:
            arrow(ax, s, [.9, 0, 1.2], [2.35, 0, 1.2], color=BLUE)
            callout(ax, s, '重新形成圆弧搭管', [1.65, 0, .66], (.44, .90), BLUE)
    save(fig, 'fig_b_flange_crossing_3d')


def bend_path():
    a = np.linspace(-np.pi/2, 0, 65)
    arc = np.column_stack([-.8+1.6*np.cos(a), 1.6+1.6*np.sin(a), np.zeros(len(a))])
    return np.vstack([straight([-2.5, 0, 0], [-.8, 0, 0], 24), arc[1:],
                      straight([.8, 1.6, 0], [.8, 2.8, 0], 20)[1:]])


def route_pose(path, fraction, radius=.55, transition=False):
    p = np.asarray(path)
    i = int(fraction*(len(p)-1))
    tangent = p[min(i+1,len(p)-1)]-p[max(0,i-1)]
    return saddle(p[i], tangent, radius=radius, transition=transition)


def sharp_elbow_scene():
    s = Scene((4.4, -10, 6.3))
    s.sharp_elbow()
    return s


def fig_c():
    fig, axs = figure('c  弯管：近 90° 急弯与连续柔弯的直接滚动',
                       '管道实体均连续｜“非连续”表示转向急变；L 型急弯保持圆弧滚动直接通过，全程不做步态变换。',
                       2, 2, [('L 型急弯 ① 滚动接近转角', '转角处管体连续，管轴方向近似发生 90° 急转'),
                              ('L 型急弯 ② 直接滚过转角', '沿转角后的管段继续圆弧滚动，全程不做步态变换'),
                              ('连续柔和弯管 ① 进入弯段', '保持圆弧搭管，开始调整局部构型朝向'),
                              ('连续柔和弯管 ② 沿弯轴推进', '圆弧包绕平面随局部管轴转动，保持目标步态')])
    for k in (0,1):
        ax=axs[k]
        s=sharp_elbow_scene()
        if k==0:
            p, st, core=saddle([-1.30,0,0],radius=.43)
        else:
            # Same saddle geometry and control partition before and after the corner;
            # only placement and orientation follow the next pipe axis.
            p, st, core=saddle([0,1.35,0],tangent=(0,1,0),radius=.43)
        s.robot(p,st)
        s.draw(ax,(-2.9,2.35,-1.95,1.74))
        callout(ax,s,'实体连续的 90° 急弯',[0,0,.43],(.68,.24))
        if k==0:
            arrow(ax,s,[-1.5,0,1.0],[-.52,0,1.0],color=BLUE)
            callout(ax,s,'转角后的管段',[0,2.1,.43],(.58,.91))
        else:
            callout(ax,s,'保持同一圆弧滚动构型',core[54],(.03,.88),BLUE)
            arrow(ax,s,[-.66,0,.92],[0,1.38,.92],color=BLUE,rad=-.22)
            arrow(ax,s,[0,1.42,.82],[0,1.98,.82],color=BLUE)
    path=bend_path()
    for k,frac in [(2,.17),(3,.68)]:
        ax=axs[k]
        s=Scene((4.4,-10,6.3))
        s.pipe(path,radius=.43)
        p,st,core=route_pose(path,frac,radius=.43)
        s.robot(p,st)
        s.draw(ax,(-2.9,2.35,-1.95,1.74))
        callout(ax,s,'连续柔弯',path[60]+[0,0,.43],(.54,.08))
        if k==2:
            arrow(ax,s,[-1.95,0,1.05],[-.70,0,1.05],color=BLUE)
        else:
            callout(ax,s,'随局部管轴改变圆弧朝向',core[54],(.02,.88),BLUE)
            arrow(ax,s,[.74,1.02,.82],[.82,1.98,.82],color=BLUE,rad=-.15)
    save(fig,'fig_c_elbow_crossing_3d')


def fig_d():
    fig,axs=figure('d  变径弯管：包绕半径调整与局部使能窗口',
                    '三维侧视｜同一长度的机器人：管径增大时，顶部包绕段占用更多连杆，悬垂段相应缩短。',
                    1,3,[('① 小径段','较小包绕半径 R1\n顶部接触段使能'),
                         ('② 渐扩弯段','局部顺应调整半径\n使能集合随接触变化更新'),
                         ('③ 大径段','较大包绕半径 R2\n更多关节参与接触包绕')],figsize=(17.4,8.4))
    path=bend_path()
    d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    u=np.clip((d-.6)/(d[-1]-1.1),0,1)
    radii=.36+.31*(u*u*(3-2*u))
    for k,frac in enumerate((.16,.52,.87)):
        ax=axs[k]
        s=Scene((4.4,-10,6.3))
        s.pipe(path,radius=radii)
        i=int(frac*(len(path)-1))
        tangent=path[min(i+1,len(path)-1)]-path[max(0,i-1)]
        p,st,core=saddle(path[i],tangent,radius=radii[i],transition=k==1)
        s.robot(p,st)
        s.draw(ax,(-3.02,2.5,-2.35,2.2))
        callout(ax,s,'小径端',[-2.5,0,.28],(.02,.33),MUTED)
        callout(ax,s,'大径端',[.8,2.8,.48],(.88,.63),MUTED,align='right')
        callout(ax,s,'局部使能段' if k!=1 else '顺应调整 / 窗口更新',core[54],(.03,.95),BLUE if k!=1 else ORANGE)
        if k in (0,2):
            c=path[i]
            arrow(ax,s,c,c+[0,0,radii[i]+BODY_R],color=GREEN,style='<->',lw=1.25)
            text_at(ax,s,'$R_1$' if k==0 else '$R_2$',c+[.23,0,.23],color=GREEN,size=11)
        else:
            arrow(ax,s,path[i]+[0,0,.95],path[min(i+24,len(path)-1)]+[0,0,.95],color=ORANGE,rad=-.08)
    save(fig,'fig_d_variable_radius_3d')


def tee_scene(down=False):
    s=Scene((3.3,-10,5.5))
    if down:
        s.pipe([0,0,-.04],[0,0,-2.2],.48)
    else:
        s.pipe([0,-.04,0],[0,-2.45,0],.48)
    s.pipe([-2.65,0,0],[2.65,0,0])
    return s


def fig_e_down():
    fig,axs=figure('e①  向下凸出 T 型管：直接通过的研究设想',
                    '三维侧视｜顶部圆弧与下侧分支的相对位置可见；两侧悬垂段的扫掠碰撞仍需检查。',
                    1,2,[('① 沿主管接近下侧分支','以顶部接触段支撑，保持圆弧搭管形态'),
                         ('② 跨过交汇区，沿主管继续','目标是保持圆弧滚动，不另增缠绕越障阶段')])
    for k,x in enumerate((-1.4,1.35)):
        s=tee_scene(True)
        p,st,_=saddle([x,0,0])
        s.robot(p,st)
        s.draw(axs[k],(-3.15,3.15,-2.65,1.62))
        callout(axs[k],s,'向下凸出的侧支',[0,-.3,-1.65],(.70,.16))
        arrow(axs[k],s,[x-.6,0,1.2],[x+.65,0,1.2],color=BLUE)
        if k==1:
            callout(axs[k],s,'顶部圆弧继续包绕',[x,0,.66],(.42,.90),BLUE)
    save(fig,'fig_e1_downward_tee_3d')


def fig_e_cross():
    fig,axs=figure('e①  侧向凸出 T 型管：单圈解旋与抬尾越障',
                    '三维侧视｜画出单侧 T 型支管；左侧与右侧情况可镜像。此路线的目标是沿主管直行。',
                    2,2,[('① 滚动接近侧支','侧支可能阻挡悬垂段，需要准备越障'),
                         ('② 形成重力矩单圈构型','由圆弧构型转为单圈包绕，为解旋做准备'),
                         ('③ 单圈解旋，尾部抬过侧支','示意前部支撑与尾部的空间避让；抬尾控制待设计'),
                         ('④ 回到主管上的圆弧滚动','越过侧向障碍，沿原主管继续推进')])
    for k,ax in enumerate(axs):
        s=tee_scene()
        if k==0:
            p,st,_=saddle([-1.5,0,0])
        elif k==1:
            p,st=coil(-.85)
        elif k==2:
            p,st,lift=lifted_transfer(.95)
        else:
            p,st,_=saddle([1.5,0,0])
        s.robot(p,st)
        s.draw(ax,(-3.10,3.10,-2.20,1.58))
        if k==0:
            callout(ax,s,'侧向凸出支管',[0,-1.70,.44],(.62,.26))
            arrow(ax,s,[-2.0,0,1.1],[-.85,0,1.1],color=BLUE)
        elif k==1:
            callout(ax,s,'重力矩单圈包绕',[-.85,0,.66],(.03,.89),BLUE)
        elif k==2:
            callout(ax,s,'尾部从侧支上方跨过',lift[53],(.06,.95),ORANGE)
            arrow(ax,s,[-.60,-.8,1.25],[.44,-.8,1.25],color=ORANGE,rad=-.12)
            callout(ax,s,'前部接触支撑',[1.0,0,.66],(.68,.27),BLUE)
        else:
            arrow(ax,s,[.90,0,1.1],[2.1,0,1.1],color=BLUE)
            text_at(ax,s,'沿主管直行',[1.4,0,1.44],color=BLUE)
    save(fig,'fig_e2_lateral_tee_crossing_3d')


def branch_transfer():
    _,_,src=saddle([-.86,0,0])
    # Source wrap opens toward the selected branch. A second, partial wrap shows
    # target-side acquisition without assigning an additional tail-lifting action.
    src=src[::-1][:80]
    _,_,dst=saddle([0,-1.40,0],tangent=(0,-1,0),radius=.48)
    dst=dst[25:]
    bridge=bezier([src[-1],[-.76,-.70,.72],[-.45,-.90,.84],dst[0]],45)
    core=np.vstack([src,bridge[1:],dst[1:]])
    remain=max(.4,BODY_LENGTH-length(core))
    first=straight(core[0]-[0,0,remain/2],core[0],25)
    last=straight(core[-1],core[-1]-[0,0,remain/2],25)
    p=np.vstack([first,core[1:],last[1:]])
    a=length(first);b=a+length(src);c=b+length(bridge);d=a+length(core)
    return p,lambda x: PURPLE if x<a or x>d else (ORANGE if b<x<c else BLUE),bridge


def fig_e_select():
    fig,axs=figure('e②  侧向 T 型管的分支选择：解旋后转向侧支',
                    '三维侧视｜目标由主管直行改为转入所选侧支；此方案不另设抬尾越障动作。',
                    1,3,[('① 单圈形态 / 选择侧支','在交汇区前形成重力矩构型'),
                         ('② 解旋，转移到目标侧支','源侧包绕逐步释放\n目标侧接触逐步建立'),
                         ('③ 在侧支上恢复圆弧','圆弧朝向跟随侧支管轴\n沿所选方向继续推进')],figsize=(17.4,8.4))
    for k,ax in enumerate(axs):
        s=tee_scene()
        if k==0:
            p,st=coil(-.85)
        elif k==1:
            p,st,bridge=branch_transfer()
        else:
            p,st,_=saddle([0,-1.55,0],tangent=(0,-1,0),radius=.48)
        s.robot(p,st)
        s.draw(ax,(-3.2,3.2,-2.3,2.0))
        if k==0:
            callout(ax,s,'所选侧支',[0,-1.95,.45],(.69,.27),GREEN)
            arrow(ax,s,[-.75,-.20,1.03],[0,-1.65,1.03],color=GREEN,rad=-.3)
        elif k==1:
            callout(ax,s,'解旋 / 转向过渡',bridge[23],(.03,.90),ORANGE)
            callout(ax,s,'目标侧包绕',[0,-1.4,.58],(.70,.21),BLUE)
        else:
            arrow(ax,s,[0,-.92,1.0],[0,-2.24,1.0],color=GREEN)
            callout(ax,s,'侧支上的圆弧构型',[0,-1.55,.58],(.04,.88),BLUE)
    save(fig,'fig_e3_tee_branch_selection_3d')


FIGURES={'a':fig_a,'b':fig_b,'c':fig_c,'d':fig_d,'e1':fig_e_down,'e2':fig_e_cross,'e3':fig_e_select}
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('figures',nargs='*',help='a b c d e1 e2 e3; omit to draw all')
    selected=parser.parse_args().figures or list(FIGURES)
    for key in selected:
        if key not in FIGURES:
            parser.error('unknown figure: '+key)
        FIGURES[key]()
