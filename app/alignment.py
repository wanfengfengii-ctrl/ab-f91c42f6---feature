"""两芯联合树轮对齐核心算法。

同一根梁材的两支树芯必须落入同一地区年表，并保证两支末环对应同一日历年。

单支树芯在“样芯环 × 年表项”网格上做路径搜索：

* 匹配 (i,j) -> (i+1,j+1)：样芯第 i 环对入年表第 j 项，差值不得越过该环容差；
* 伪环 (i,j) -> (i+1,j)   ：跳过一个样芯环（年表中没有对应年）；
* 缺环 (i,j) -> (i,j+1)   ：跳过一个年表项（该年样芯缺轮）。

路径第一步与最后一步都必须是匹配，因此样芯只是放在年表窗口内，
年表前后多余项不会被误记为缺环；所有跳过都严格位于已匹配环之间。

一条完整映射也可等价表述为一串严格双增的匹配对
    (0, j0) < (i1, j1) < ... < (n-1, e)
相邻匹配对之间的样芯环即伪环、年表项即缺环。

优化按字典序最小化：
    1. 两芯缺环与伪环总数；
    2. 全部匹配差值的最大绝对值；
    3. 全部匹配差值绝对值总和。

实现分两阶段：
    A. 帕累托动态规划求每个终年的最优指标（同指标向量只留一条代表，
       指标值与映射数量无关，剪枝不影响最优值）；
    B. 固定最优目标后，用记忆化 DFS 按映射序精确枚举前两份同优见证，
       因此“唯一 / 歧义”的判定是精确的。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# 阶段 A：帕累托动态规划（求指标，不求全部映射）
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _Rec:
    miss: int
    fls: int
    mmax: int
    dsum: int
    last: str                 # 'M' / 'F' / 'X'


def _v(r: _Rec) -> tuple[int, int, int, int]:
    return (r.miss, r.fls, r.mmax, r.dsum)


def _add(bucket: list[_Rec], r: _Rec) -> None:
    """并入同一网格状态：严格支配者丢弃；指标向量完全相同只留一条代表。"""
    rv = _v(r)
    for q in bucket:
        qv = _v(q)
        if qv[0] <= rv[0] and qv[1] <= rv[1] and qv[2] <= rv[2] and qv[3] <= rv[3]:
            return
    bucket.append(r)


def pareto_fronts(
    sample: list[int],
    chron: list[int],
    tol: list[int],
    miss_cap: int,
    false_cap: int,
) -> list[list[list[_Rec]]]:
    """返回 dp[i][j]：处理完前 i 个样芯环、年表推进到 j 的帕累托前沿。"""
    n, m = len(sample), len(chron)
    dp: list[list[list[_Rec]]] = [[[] for _ in range(m + 1)] for _ in range(n + 1)]

    # 首环必须匹配，不允许开头跳过。
    for s in range(m):
        d = abs(sample[0] - chron[s])
        if d <= tol[0]:
            _add(dp[1][s + 1], _Rec(0, 0, d, d, "M"))

    for i in range(1, n):
        for j in range(m + 1):
            for r in tuple(dp[i][j]):
                # 伪环：跳过样芯第 i 环
                if r.fls < false_cap:
                    _add(dp[i + 1][j],
                         _Rec(r.miss, r.fls + 1, r.mmax, r.dsum, "F"))
                if j < m:
                    d = abs(sample[i] - chron[j])
                    # 缺环：跳过年表第 j 项
                    if r.miss < miss_cap:
                        _add(dp[i][j + 1],
                             _Rec(r.miss + 1, r.fls, r.mmax, r.dsum, "X"))
                    # 匹配
                    if d <= tol[i]:
                        _add(dp[i + 1][j + 1],
                             _Rec(r.miss, r.fls, max(r.mmax, d), r.dsum + d,
                                  "M"))
    return dp


def end_records(dp: list[list[list[_Rec]]], n: int,
                min_matches: int) -> dict[int, list[_Rec]]:
    """终年表下标 -> 末步为匹配且匹配数达标的记录。"""
    out: dict[int, list[_Rec]] = {}
    for j, bucket in enumerate(dp[n]):
        rs = [r for r in bucket if r.last == "M" and n - r.fls >= min_matches]
        if rs:
            out[j - 1] = rs
    return out


def core_front(
    sample: list[int], chron: list[int], tol: list[int],
    miss_cap: int, false_cap: int, min_matches: int = 8,
) -> dict[int, list[_Rec]]:
    dp = pareto_fronts(sample, chron, tol, miss_cap, false_cap)
    return end_records(dp, len(sample), min_matches)


# --------------------------------------------------------------------------- #
# 阶段 B：固定目标的精确见证枚举
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Mapping:
    pairs: tuple[tuple[int, int, int], ...]   # (样芯下标, 年表下标, 差值)
    missing: tuple[int, ...]
    false: tuple[int, ...]

    @property
    def miss(self) -> int:
        return len(self.missing)

    @property
    def fls(self) -> int:
        return len(self.false)

    @property
    def skips(self) -> int:
        return self.miss + self.fls

    @property
    def mmax(self) -> int:
        return max(d for _, _, d in self.pairs)

    @property
    def dsum(self) -> int:
        return sum(d for _, _, d in self.pairs)

    @property
    def sig(self) -> tuple[tuple[int, int], ...]:
        return tuple((i, j) for i, j, _ in self.pairs)


class _Enumerator:
    """在固定终年、跳环总数、差值上限 / 差值总和目标下枚举映射。

    映射即一串严格双增的匹配对，首对样芯下标为 0、末对样芯下标为 n-1
    且年表下标为 end。按匹配对字典序（即“映射序”）产出。
    """

    def __init__(self, sample, chron, tol, end, skips, mmax_cap,
                 miss_cap, false_cap):
        self.s = sample
        self.c = chron
        self.t = tol
        self.n = len(sample)
        self.end = end
        self.skips = skips
        self.M = mmax_cap
        self.miss_cap = miss_cap
        self.false_cap = false_cap

    @lru_cache(maxsize=None)
    def _feasible(self, i: int, j: int, sk_left: int,
                  dsum_left: Optional[int]) -> bool:
        """从“下一对样芯下标 >= i、年表下标 >= j”出发能否合法收尾。"""
        if sk_left < 0 or i > self.n - 1 or j > self.end:
            return False
        rings_left = self.n - i
        span = self.end - j
        # 剩余 f 个伪环、x 个缺环且 f+x=sk_left：
        # f ≤ rings_left-1（至少末环要匹配）、x ≤ span（至少一个年表槽匹配），
        # 有解当且仅当 span ≥ sk_left-(rings_left-1)。
        if span < sk_left - (rings_left - 1):
            return False
        # 收尾选项：下一对即末环 (n-1, end)，样芯 i..n-2 全作伪环、
        # 年表 j..end-1 全作缺环（末环紧邻前一跳环也由此覆盖）。
        f_last = self.n - 1 - i
        x_last = self.end - j
        if f_last + x_last == sk_left and f_last <= self.false_cap \
                and x_last <= self.miss_cap:
            d = abs(self.s[self.n - 1] - self.c[self.end])
            if d <= self.t[self.n - 1] and d <= self.M and (
                    dsum_left is None or d == dsum_left):
                return True
        if i == self.n - 1:
            return False
        # 枚举下一对 (ip, jp)：跳过样芯 i..ip-1（伪环）与年表 j..jp-1（缺环）
        max_ip = min(self.n - 2, i + sk_left)
        for ip in range(i, max_ip + 1):
            f = ip - i
            rem = sk_left - f
            jp_hi = min(self.end - 1, j + rem)
            for jp in range(j, jp_hi + 1):
                used = f + (jp - j)
                d = abs(self.s[ip] - self.c[jp])
                if d > self.t[ip] or d > self.M:
                    continue
                ndl = None if dsum_left is None else dsum_left - d
                if ndl is not None and ndl < 0:
                    continue
                # 伪环与缺环的分顶限额只在整体计数时校验：
                # 累计伪环 / 缺环数无法仅由 sk_left 表达，保守放行，
                # 由产出端重建映射后复核（限额 ≤3，额外搜索量很小）。
                if self._feasible(ip + 1, jp + 1, sk_left - used, ndl):
                    return True
        return False

    def dfs_yield(self, i, j, sk_left, dsum_left, pairs, missing, false):
        """按映射序惰性产出；dsum_left 给定时只产出差值和恰为目标值的映射。"""
        f_last = self.n - 1 - i
        x_last = self.end - j
        can_close = (
            f_last + x_last == sk_left
            and len(false) + f_last <= self.false_cap
            and len(missing) + x_last <= self.miss_cap
        )
        d_close = abs(self.s[self.n - 1] - self.c[self.end])
        close_ok = (can_close and d_close <= self.t[self.n - 1]
                    and d_close <= self.M
                    and (dsum_left is None or d_close == dsum_left))
        # 映射序：下一对 (ip, jp)（ip ≤ n-2）一律早于直接收尾对
        # (n-1, end)，故先枚举中间对、最后产出直接收尾。
        if i < self.n - 1:
            max_ip = min(self.n - 2, i + sk_left)
            for ip in range(i, max_ip + 1):
                f_skip = ip - i
                if len(false) + f_skip > self.false_cap:
                    continue
                jp_hi = min(self.end - 1, j + (sk_left - f_skip))
                for jp in range(j, jp_hi + 1):
                    x_skip = jp - j
                    used = f_skip + x_skip
                    if used > sk_left:
                        continue
                    if len(missing) + x_skip > self.miss_cap:
                        continue
                    d = abs(self.s[ip] - self.c[jp])
                    if d > self.t[ip] or d > self.M:
                        continue
                    ndl = None if dsum_left is None else dsum_left - d
                    if ndl is not None and ndl < 0:
                        continue
                    if not self._feasible(ip + 1, jp + 1,
                                          sk_left - used, ndl):
                        continue
                    pairs.append((ip, jp, d))
                    yield from self.dfs_yield(
                        ip + 1, jp + 1, sk_left - used, ndl, pairs,
                        missing + list(range(j, jp)),
                        false + list(range(i, ip)))
                    pairs.pop()

        # 直接收尾：末环匹配 end，其间样芯环为伪环、年表项为缺环。
        if close_ok:
            pairs.append((self.n - 1, self.end, d_close))
            mp = Mapping(
                tuple(pairs),
                tuple(sorted(missing + list(range(j, self.end)))),
                tuple(sorted(false + list(range(i, self.n - 1)))),
            )
            if (mp.miss <= self.miss_cap and mp.fls <= self.false_cap
                    and len(pairs) >= 8):
                yield mp
            pairs.pop()

    def iter_mappings(self, dsum_target, limit=None):
        """从首环对起按映射序惰性产出（至多 limit 份）。"""
        emitted = 0
        for j0 in range(0, self.end + 1):
            d = abs(self.s[0] - self.c[j0])
            if d > self.t[0] or d > self.M:
                continue
            ndl = None if dsum_target is None else dsum_target - d
            if ndl is not None and ndl < 0:
                continue
            if not self._feasible(1, j0 + 1, self.skips, ndl):
                continue
            for mp in self.dfs_yield(1, j0 + 1, self.skips, ndl,
                                     [(0, j0, d)], [], []):
                yield mp
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def _enumerate_simple(sample, chron, tol, end, skips, mmax_cap, dsum_target,
                      miss_cap, false_cap, limit=None):
    """枚举入口；首对之前不允许跳过，跳环全部发生在匹配对之间。

    返回按映射序排列的生成器；limit 给定时至多产出 limit 份。
    """
    en = _Enumerator(sample, chron, tol, end, skips, mmax_cap,
                     miss_cap, false_cap)
    return en.iter_mappings(dsum_target, limit)


# --------------------------------------------------------------------------- #
# 表层残失环（terminal_loss）支持
#
# 梁材表层常因刨削损失少量边材环：每支末环仍真实匹配年表（下标 e），
# 但砍伐年下标为 f = e + l，l 为该芯选定的残失环数（0–3）。两芯须以
# 各自的 (e, l) 落到同一 f。残失环既不是缺环也不是伪环，不进入三级目标；
# 但不同 (f, l) 组合是不同解释，全部同优解释都参与唯一 / 歧义判定。
# --------------------------------------------------------------------------- #

def _felling_table(
    front: dict[int, list[_Rec]], lo: int, hi: int,
) -> dict[int, list[tuple[int, int, _Rec]]]:
    """砍伐年表下标 f -> [(残失 l, 实际终年表下标 e, 记录 r), ...]。"""
    table: dict[int, list[tuple[int, int, _Rec]]] = {}
    for e, rs in front.items():
        for l in range(lo, hi + 1):
            table.setdefault(e + l, []).extend((l, e, r) for r in rs)
    return table


def _joint_objective_loss(
    fa: dict[int, list[_Rec]],
    fb: dict[int, list[_Rec]],
    loss_a: tuple[int, int],
    loss_b: tuple[int, int],
) -> Optional[dict[str, Any]]:
    """跨砍伐年求 (总跳环, 最大差值, 差值和) 最优值与可达砍伐年集合。

    与无残失版本结构一致，只是终年由 e 换成 f=e+l，残失不参与任何目标。
    """
    ta = _felling_table(fa, *loss_a)
    tb = _felling_table(fb, *loss_b)
    shared = set(ta) & set(tb)
    if not shared:
        return None

    def _k(opts: list[tuple[int, int, _Rec]]) -> int:
        return min(r.miss + r.fls for _, _, r in opts)

    ka = {f: _k(opts) for f, opts in ta.items()}
    kb = {f: _k(opts) for f, opts in tb.items()}

    s_total = min(ka[f] + kb[f] for f in shared)
    fellings = [f for f in shared if ka[f] + kb[f] == s_total]

    mstar: Optional[int] = None
    sstar: Optional[int] = None
    per_f: dict[int, tuple[int, int]] = {}
    for f in fellings:
        pa = [x for x in ta[f] if x[2].miss + x[2].fls == ka[f]]
        pb = [x for x in tb[f] if x[2].miss + x[2].fls == kb[f]]
        m_f = max(min(r.mmax for _, _, r in pa),
                  min(r.mmax for _, _, r in pb))
        d_a = min(r.dsum for _, _, r in pa if r.mmax <= m_f)
        d_b = min(r.dsum for _, _, r in pb if r.mmax <= m_f)
        per_f[f] = (m_f, d_a + d_b)
        if mstar is None or (m_f, d_a + d_b) < (mstar, sstar):
            mstar, sstar = m_f, d_a + d_b
    best_fellings = [f for f in fellings if per_f[f] == (mstar, sstar)]
    return {"skips": s_total, "mmax": mstar, "dsum": sstar,
            "fellings": sorted(best_fellings),
            "ka": ka, "kb": kb, "ta": ta, "tb": tb}


def find_witnesses_loss(sa, sb, chron, tol_a, tol_b, fa, fb,
                        miss_caps, false_caps, opt,
                        loss_a, loss_b) -> list[tuple[int, int, int,
                                                      Mapping, Mapping]]:
    """按 (砍伐年, A 残失, B 残失, A 映射序, B 映射序) 取前两份联合见证。

    返回元组 (f, l_a, l_b, Mapping_a, Mapping_b)；f 可超出年表末项
    （砍伐发生在年表窗口之后），但枚举仍只在实际终年 e 上进行。
    """
    mstar, sstar = opt["mmax"], opt["dsum"]
    lo_a, hi_a = loss_a
    lo_b, hi_b = loss_b
    out: list[tuple[int, int, int, Mapping, Mapping]] = []
    for f in opt["fellings"]:
        k_a, k_b = opt["ka"][f], opt["kb"][f]
        for l_a in range(lo_a, hi_a + 1):
            e_a = f - l_a
            bucket_a = fa.get(e_a)
            if not bucket_a or not any(
                    r.miss + r.fls == k_a for r in bucket_a):
                continue
            for l_b in range(lo_b, hi_b + 1):
                e_b = f - l_b
                bucket_b = fb.get(e_b)
                if not bucket_b or not any(
                        r.miss + r.fls == k_b for r in bucket_b):
                    continue

                def b_for(dsum_need: int, _e=e_b, _k=k_b) -> list[Mapping]:
                    return list(_enumerate_core(
                        sb, chron, tol_b, _e, _k, mstar,
                        miss_caps[1], false_caps[1], dsum_need, 2))

                found = 0
                for ma in _enumerate_core(sa, chron, tol_a, e_a, k_a, mstar,
                                          miss_caps[0], false_caps[0],
                                          None, None):
                    for mb in b_for(sstar - ma.dsum):
                        out.append((f, l_a, l_b, ma, mb))
                        found += 1
                        if found == 2:
                            break
                    if found == 2:
                        break
                if len(out) >= 2:
                    return out
    return out


# --------------------------------------------------------------------------- #
# 联合最优选择
# --------------------------------------------------------------------------- #

def _min_skips(front: dict[int, list[_Rec]]) -> dict[int, int]:
    return {e: min(r.miss + r.fls for r in rs) for e, rs in front.items()}


def _joint_objective(fa: dict[int, list[_Rec]],
                     fb: dict[int, list[_Rec]]) -> Optional[dict[str, Any]]:
    """跨终年求 (总跳环, 最大差值, 差值和) 最优值与可达终年集合。"""
    ka, kb = _min_skips(fa), _min_skips(fb)
    shared = set(ka) & set(kb)
    if not shared:
        return None
    s_total = min(ka[e] + kb[e] for e in shared)
    ends = [e for e in shared if ka[e] + kb[e] == s_total]

    mstar: Optional[int] = None
    sstar: Optional[int] = None
    per_end: dict[int, tuple[int, int]] = {}
    for e in ends:
        pa = [r for r in fa[e] if r.miss + r.fls == ka[e]]
        pb = [r for r in fb[e] if r.miss + r.fls == kb[e]]
        m_e = max(min(r.mmax for r in pa), min(r.mmax for r in pb))
        # 联合最大差值恰为 m_e 时，两侧差值和各自可取的最小值。
        d_a = min(r.dsum for r in pa if r.mmax <= m_e)
        d_b = min(r.dsum for r in pb if r.mmax <= m_e)
        per_end[e] = (m_e, d_a + d_b)
        if mstar is None or (m_e, d_a + d_b) < (mstar, sstar):
            mstar, sstar = m_e, d_a + d_b
    best_ends = [e for e in ends if per_end[e] == (mstar, sstar)]
    return {"skips": s_total, "mmax": mstar, "dsum": sstar,
            "ends": sorted(best_ends), "ka": ka, "kb": kb}


def _enumerate_core(sample, chron, tol, end, skips, mstar, miss_cap, false_cap,
                    dsum_target, limit) -> list[Mapping]:
    return _enumerate_simple(sample, chron, tol, end, skips, mstar,
                             dsum_target, miss_cap, false_cap, limit)


def find_witnesses(sa, sb, chron, ta, tb, fa, fb,
                   miss_caps, false_caps, opt) -> list[tuple[Mapping, Mapping]]:
    """跨所有最优终年，按映射序取前两份联合见证。

    A 侧按映射序惰性枚举；对每个 A 映射，B 侧只枚举差值和恰为
    (sstar - A 差值和) 的前两份。某终年产出两份即够用于判定歧义，
    但为给出跨终年全局映射序的前两份，每个终年各取至多两份后归并。
    """
    mstar, sstar = opt["mmax"], opt["dsum"]
    pool: list[tuple[Mapping, Mapping]] = []
    for e in opt["ends"]:
        b_cache: dict[int, list[Mapping]] = {}

        def b_for(dsum_need: int) -> list[Mapping]:
            if dsum_need not in b_cache:
                b_cache[dsum_need] = list(_enumerate_core(
                    sb, chron, tb, e, opt["kb"][e], mstar,
                    miss_caps[1], false_caps[1], dsum_need, 2))
            return b_cache[dsum_need]

        found_here = 0
        # A 侧为惰性生成器，凑满两份配对即停止枚举。
        for ma in _enumerate_core(sa, chron, ta, e, opt["ka"][e], mstar,
                                  miss_caps[0], false_caps[0], None, None):
            for mb in b_for(sstar - ma.dsum):
                pool.append((ma, mb))
                found_here += 1
                if found_here == 2:
                    break
            if found_here == 2:
                break
    pool.sort(key=lambda t: (t[0].sig, t[1].sig))
    return pool[:2]


# --------------------------------------------------------------------------- #
# 响应组装
# --------------------------------------------------------------------------- #

def _mapping_payload(mp: Mapping, start_year: int, sample: list[int],
                     chron: list[int],
                     outer_loss: Optional[int] = None) -> dict[str, Any]:
    payload = {
        "matched": [
            {
                "sample_ring": si + 1,
                "chronology_index": cj,
                "year": start_year + cj,
                "sample_value": sample[si],
                "chronology_value": chron[cj],
                "difference": d,
            }
            for si, cj, d in mp.pairs
        ],
        "missing_chronology_years": [start_year + cj for cj in mp.missing],
        "missing_chronology_indices": list(mp.missing),
        "false_sample_rings": [si + 1 for si in mp.false],
        "missing_count": mp.miss,
        "false_count": mp.fls,
        "match_count": len(mp.pairs),
        "end_year": start_year + mp.pairs[-1][1],
        "max_difference": mp.mmax,
        "difference_sum": mp.dsum,
    }
    # 未启用 terminal_loss 时不输出该字段，保持既有响应形态不变。
    if outer_loss is not None:
        payload["outer_loss"] = outer_loss
    return payload


def _witness_payload(ma: Mapping, mb: Mapping, start_year: int,
                     sa: list[int], sb: list[int],
                     chron: list[int],
                     loss: Optional[tuple[int, int]] = None,
                     felling_year: Optional[int] = None) -> dict[str, Any]:
    e = ma.pairs[-1][1]
    if loss is None:
        return {
            "common_end_year": start_year + e,
            "mappings": {
                "a": _mapping_payload(ma, start_year, sa, chron),
                "b": _mapping_payload(mb, start_year, sb, chron),
            },
        }
    la, lb = loss
    return {
        "common_end_year": felling_year,
        "mappings": {
            "a": _mapping_payload(ma, start_year, sa, chron, la),
            "b": _mapping_payload(mb, start_year, sb, chron, lb),
        },
    }


# --------------------------------------------------------------------------- #
# 无解证据
# --------------------------------------------------------------------------- #

def _relaxed_info(sample, chron, tol, miss_cap, false_cap,
                  min_matches) -> tuple[dict[int, list[_Rec]], Optional[int]]:
    front = core_front(sample, chron, tol, miss_cap, false_cap, min_matches)
    k = None
    if front:
        k = min(min(r.miss + r.fls for r in rs) for rs in front.values())
    return front, k


def _best_by_end(front: dict[int, list[_Rec]]) -> dict[int, _Rec]:
    best: dict[int, _Rec] = {}
    for e, rs in front.items():
        best[e] = min(rs, key=lambda r: (r.miss + r.fls, r.mmax, r.dsum))
    return best


def _no_solution_payload(start_year, chron, sa, sb, ta, tb,
                         caps, fa, fb) -> dict[str, Any]:
    (miss_a, false_a), (miss_b, false_b) = caps
    n = (len(sa), len(sb))
    m = len(chron)

    rel8_a, k8_a = _relaxed_info(sa, chron, ta, m, n[0] - 8, 8)
    rel8_b, k8_b = _relaxed_info(sb, chron, tb, m, n[1] - 8, 8)
    rel1_a, k1_a = _relaxed_info(sa, chron, ta, m, n[0] - 1, 1)
    rel1_b, k1_b = _relaxed_info(sb, chron, tb, m, n[1] - 1, 1)

    last_a = [j for j in range(m) if abs(sa[-1] - chron[j]) <= ta[-1]]
    last_b = [j for j in range(m) if abs(sb[-1] - chron[j]) <= tb[-1]]

    shared_relaxed = sorted(set(rel1_a) & set(rel1_b))

    if k1_a is None or k1_b is None:
        reason = "last_ring_unmatchable"
    elif k8_a is None or k8_b is None:
        reason = "insufficient_matchable_rings"
    elif not fa or not fb:
        reason = "skip_limits_exceeded"
    elif not shared_relaxed:
        reason = "no_common_end_year"
    else:
        reason = "common_end_requires_more_skips"

    def core_ev(rel8, rel1, k8, last, strict, miss_cap, false_cap, nn):
        return {
            "feasible_within_limits": bool(strict),
            "last_ring_candidate_years": [start_year + j for j in last],
            "reachable_end_years_within_limits": [
                start_year + e for e in sorted(strict)
            ],
            "min_skips_within_limits": (
                min(min(r.miss + r.fls for r in rs)
                    for rs in strict.values()) if strict else None
            ),
            "min_skips_with_eight_matches": k8,
            "min_skips_with_any_match": (
                min(min(r.miss + r.fls for r in rs)
                    for rs in rel1.values()) if rel1 else None
            ),
            "limits": {"missing": miss_cap, "false": false_cap},
        }

    ev_a = core_ev(rel8_a, rel1_a, k8_a, last_a, fa, miss_a, false_a, n[0])
    ev_b = core_ev(rel8_b, rel1_b, k8_b, last_b, fb, miss_b, false_b, n[1])

    best8 = (_best_by_end(rel8_a), _best_by_end(rel8_b))
    best1 = (_best_by_end(rel1_a), _best_by_end(rel1_b))
    shared_details = []
    for j in shared_relaxed:
        entry: dict[str, Any] = {
            "year": start_year + j, "chronology_index": j,
        }
        for idx, (miss_cap, false_cap) in enumerate(
            ((miss_a, false_a), (miss_b, false_b))
        ):
            r = best8[idx].get(j) or best1[idx].get(j)
            if r is None:
                entry["ab"[idx]] = None
                continue
            entry["ab"[idx]] = {
                "missing": r.miss,
                "false": r.fls,
                "total_skips": r.miss + r.fls,
                "match_count": n[idx] - r.fls,
                "max_difference": r.mmax,
                "difference_sum": r.dsum,
                "within_limits": (
                    r.miss <= miss_cap and r.fls <= false_cap
                    and n[idx] - r.fls >= 8
                ),
            }
        shared_details.append(entry)

    return {
        "status": "no_solution",
        "common_end_year": None,
        "mappings": None,
        "objectives": None,
        "witnesses": [],
        "evidence": {
            "reason": reason,
            "shared_candidate_years": [start_year + j for j in shared_relaxed],
            "cores": {"a": ev_a, "b": ev_b},
            "shared_end_details": shared_details,
        },
    }


# --------------------------------------------------------------------------- #
# 顶层入口
# --------------------------------------------------------------------------- #

def _inferred_years(front_keys, lo: int, hi: int, start_year: int) -> list[int]:
    """由可达终年表下标集合与残失区间推出所有可达砍伐年（日历年）。"""
    years = {start_year + e + l
             for e in front_keys for l in range(lo, hi + 1)}
    return sorted(years)


def _no_solution_payload_loss(start_year, chron, sa, sb, ta, tb,
                              caps, fa, fb, loss_a, loss_b) -> dict[str, Any]:
    """残失模式无解：列出双方各自可达的推定砍伐年与限额诊断。"""
    (miss_a, false_a), (miss_b, false_b) = caps
    n = (len(sa), len(sb))
    m = len(chron)
    lo_a, hi_a = loss_a
    lo_b, hi_b = loss_b

    rel8_a, k8_a = _relaxed_info(sa, chron, ta, m, n[0] - 8, 8)
    rel8_b, k8_b = _relaxed_info(sb, chron, tb, m, n[1] - 8, 8)
    rel1_a, k1_a = _relaxed_info(sa, chron, ta, m, n[0] - 1, 1)
    rel1_b, k1_b = _relaxed_info(sb, chron, tb, m, n[1] - 1, 1)

    last_a = [j for j in range(m) if abs(sa[-1] - chron[j]) <= ta[-1]]
    last_b = [j for j in range(m) if abs(sb[-1] - chron[j]) <= tb[-1]]

    relax_fa = set(_felling_table(rel1_a, lo_a, hi_a))
    relax_fb = set(_felling_table(rel1_b, lo_b, hi_b))
    shared_relaxed = sorted(relax_fa & relax_fb)

    if k1_a is None or k1_b is None:
        reason = "last_ring_unmatchable"
    elif k8_a is None or k8_b is None:
        reason = "insufficient_matchable_rings"
    elif not fa or not fb:
        reason = "skip_limits_exceeded"
    elif not shared_relaxed:
        reason = "no_common_felling_year"
    else:
        reason = "common_end_requires_more_skips"

    def core_ev(rel8, k8, rel1, last, strict, lo, hi,
                miss_cap, false_cap):
        return {
            "feasible_within_limits": bool(strict),
            "last_ring_candidate_years": [start_year + j for j in last],
            "reachable_end_years_within_limits": [
                start_year + e for e in sorted(strict)
            ],
            "reachable_felling_years": _inferred_years(
                sorted(strict), lo, hi, start_year),
            "min_skips_within_limits": (
                min(min(r.miss + r.fls for r in rs)
                    for rs in strict.values()) if strict else None
            ),
            "min_skips_with_eight_matches": k8,
            "min_skips_with_any_match": (
                min(min(r.miss + r.fls for r in rs)
                    for rs in rel1.values()) if rel1 else None
            ),
            "terminal_loss": {"min": lo, "max": hi},
            "limits": {"missing": miss_cap, "false": false_cap},
        }

    ev_a = core_ev(rel8_a, k8_a, rel1_a, last_a, fa, lo_a, hi_a,
                   miss_a, false_a)
    ev_b = core_ev(rel8_b, k8_b, rel1_b, last_b, fb, lo_b, hi_b,
                   miss_b, false_b)

    best8 = (_best_by_end(rel8_a), _best_by_end(rel8_b))
    best1 = (_best_by_end(rel1_a), _best_by_end(rel1_b))
    shared_details = []
    for f in shared_relaxed:
        entry: dict[str, Any] = {
            "felling_year": start_year + f,
            "felling_chronology_index": f,
        }
        for idx, (lo, hi, miss_cap, false_cap) in enumerate((
            (lo_a, hi_a, miss_a, false_a),
            (lo_b, hi_b, miss_b, false_b),
        )):
            choice = None
            for l in range(lo, hi + 1):
                e = f - l
                r = best8[idx].get(e) or best1[idx].get(e)
                if r is None:
                    continue
                cand = (r.miss + r.fls, r.mmax, r.dsum, l, e, r)
                if choice is None or cand[:3] < choice[:3]:
                    choice = cand
            if choice is None:
                entry["ab"[idx]] = None
                continue
            _, _, _, l, e, r = choice
            entry["ab"[idx]] = {
                "outer_loss": l,
                "end_year": start_year + e,
                "chronology_index": e,
                "missing": r.miss,
                "false": r.fls,
                "total_skips": r.miss + r.fls,
                "match_count": n[idx] - r.fls,
                "max_difference": r.mmax,
                "difference_sum": r.dsum,
                "within_limits": (
                    r.miss <= miss_cap and r.fls <= false_cap
                    and n[idx] - r.fls >= 8
                ),
            }
        shared_details.append(entry)

    return {
        "status": "no_solution",
        "common_end_year": None,
        "mappings": None,
        "objectives": None,
        "witnesses": [],
        "evidence": {
            "reason": reason,
            "shared_candidate_years": [start_year + f for f in shared_relaxed],
            "cores": {"a": ev_a, "b": ev_b},
            "shared_end_details": shared_details,
        },
    }


def _align_loss(payload: dict[str, Any]) -> dict[str, Any]:
    chron: list[int] = payload["chronology"]
    start_year: int = payload["start_year"]
    sa, sb = payload["sample_a"], payload["sample_b"]
    ta, tb = payload["tolerance_a"], payload["tolerance_b"]
    miss_a, false_a = payload["max_missing_a"], payload["max_false_a"]
    miss_b, false_b = payload["max_missing_b"], payload["max_false_b"]
    caps = ((miss_a, false_a), (miss_b, false_b))
    tl = payload["terminal_loss"]
    loss_a = (tl["a"]["min"], tl["a"]["max"])
    loss_b = (tl["b"]["min"], tl["b"]["max"])

    fa = core_front(sa, chron, ta, miss_a, false_a, 8)
    fb = core_front(sb, chron, tb, miss_b, false_b, 8)

    opt = _joint_objective_loss(fa, fb, loss_a, loss_b)
    if opt is None:
        return _no_solution_payload_loss(
            start_year, chron, sa, sb, ta, tb, caps, fa, fb, loss_a, loss_b)

    raw = find_witnesses_loss(
        sa, sb, chron, ta, tb, fa, fb,
        (miss_a, miss_b), (false_a, false_b), opt, loss_a, loss_b,
    )
    # 全局排序：砍伐年、A 残失、B 残失、A 映射序、B 映射序。
    raw.sort(key=lambda t: (t[0], t[1], t[2], t[3].sig, t[4].sig))
    if not raw:
        return _no_solution_payload_loss(
            start_year, chron, sa, sb, ta, tb, caps, fa, fb, loss_a, loss_b)

    f0, la0, lb0, ma0, mb0 = raw[0]
    felling_year = start_year + f0
    status = "unique" if len(raw) == 1 else "ambiguous"
    primary = _witness_payload(ma0, mb0, start_year, sa, sb, chron,
                               loss=(la0, lb0), felling_year=felling_year)
    return {
        "status": status,
        "common_end_year": felling_year,
        "mappings": primary["mappings"],
        "objectives": {
            "total_skips": opt["skips"],
            "max_difference": opt["mmax"],
            "difference_sum": opt["dsum"],
        },
        "witnesses": [
            _witness_payload(ma, mb, start_year, sa, sb, chron,
                             loss=(la, lb),
                             felling_year=start_year + f)
            for f, la, lb, ma, mb in raw
        ],
        "evidence": None,
    }


def align_joint(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("terminal_loss") is not None:
        return _align_loss(payload)

    chron: list[int] = payload["chronology"]
    start_year: int = payload["start_year"]
    sa, sb = payload["sample_a"], payload["sample_b"]
    ta, tb = payload["tolerance_a"], payload["tolerance_b"]
    miss_a, false_a = payload["max_missing_a"], payload["max_false_a"]
    miss_b, false_b = payload["max_missing_b"], payload["max_false_b"]
    caps = ((miss_a, false_a), (miss_b, false_b))

    fa = core_front(sa, chron, ta, miss_a, false_a, 8)
    fb = core_front(sb, chron, tb, miss_b, false_b, 8)

    opt = _joint_objective(fa, fb)
    if opt is None:
        return _no_solution_payload(start_year, chron, sa, sb, ta, tb,
                                    caps, fa, fb)

    witnesses = find_witnesses(
        sa, sb, chron, ta, tb, fa, fb,
        (miss_a, miss_b), (false_a, false_b), opt,
    )
    if not witnesses:
        # 理论不可达（最优值必有映射支撑），保守按无解处理。
        return _no_solution_payload(start_year, chron, sa, sb, ta, tb,
                                    caps, fa, fb)

    ma0, mb0 = witnesses[0]
    status = "unique" if len(witnesses) == 1 else "ambiguous"
    primary = _witness_payload(ma0, mb0, start_year, sa, sb, chron)
    return {
        "status": status,
        "common_end_year": primary["common_end_year"],
        "mappings": primary["mappings"],
        "objectives": {
            "total_skips": opt["skips"],
            "max_difference": opt["mmax"],
            "difference_sum": opt["dsum"],
        },
        "witnesses": [
            _witness_payload(ma, mb, start_year, sa, sb, chron)
            for ma, mb in witnesses
        ],
        "evidence": None,
    }
