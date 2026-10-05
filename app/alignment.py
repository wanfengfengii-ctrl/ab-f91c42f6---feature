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

残失校准（可选）：请求可附带 terminal_loss，为两芯分别给出 0–3 的
边材残失环数闭区间。启用后联合条件改为“各芯末环匹配年 + 选定残失数
= 同一砍伐年”；残失环不计入缺环、伪环与原三级目标，所有同优解释
（含同一对映射的不同砍伐年取值）都参与唯一性判定，见证按
（砍伐年, A 残失数, B 残失数, 映射序）稳定排列。成功响应中
common_end_year 表示共同砍伐年，各芯映射保留实际末环年份并附
outer_loss。未提供该字段时，请求、响应、裁决与无解语义与原先一致。
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
        # 直接收尾：末环匹配 end，最后间隙的样芯环即伪环、年表项即缺环。
        # 剩余跳环数恰等于直接收尾所需时，这是唯一的收尾方式。
        if sk_left == (self.n - 1 - i) + (self.end - j):
            d = abs(self.s[self.n - 1] - self.c[self.end])
            return (d <= self.t[self.n - 1] and d <= self.M
                    and (dsum_left is None or d == dsum_left))
        if i == self.n - 1:
            return False
        rings_left = self.n - i
        span = self.end - j
        # 剩余 f 个伪环、x 个缺环且 f+x=sk_left：
        # f ≤ rings_left-1（至少末环要匹配）、x ≤ span（至少一个年表槽匹配），
        # 有解当且仅当 span ≥ sk_left-(rings_left-1)。
        if span < sk_left - (rings_left - 1):
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
        # 直接收尾：末环匹配 end，最后间隙的样芯环即伪环、年表项即缺环。
        # 剩余跳环数恰等于直接收尾所需时，这是唯一的收尾方式。
        if sk_left == (self.n - 1 - i) + (self.end - j):
            d = abs(self.s[self.n - 1] - self.c[self.end])
            if (d <= self.t[self.n - 1] and d <= self.M
                    and (dsum_left is None or d == dsum_left)):
                pairs.append((self.n - 1, self.end, d))
                mp = Mapping(
                    tuple(pairs),
                    tuple(sorted(missing + list(range(j, self.end)))),
                    tuple(sorted(false + list(range(i, self.n - 1)))))
                if (mp.miss <= self.miss_cap and mp.fls <= self.false_cap
                        and len(pairs) >= 8):
                    yield mp
                pairs.pop()
            return
        if i == self.n - 1:
            return
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
                if not self._feasible(ip + 1, jp + 1, sk_left - used, ndl):
                    continue
                pairs.append((ip, jp, d))
                yield from self.dfs_yield(
                    ip + 1, jp + 1, sk_left - used, ndl, pairs,
                    missing + list(range(j, jp)),
                    false + list(range(i, ip)))
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


def _joint_objective_loss(fa: dict[int, list[_Rec]],
                          fb: dict[int, list[_Rec]],
                          loss_a: tuple[int, int],
                          loss_b: tuple[int, int]) -> Optional[dict[str, Any]]:
    """残失校准下的联合最优：候选为可推出同一砍伐年的终年对 (e_a, e_b)。

    终年对可行当且仅当区间 [e_a+lo_a, e_a+hi_a] 与 [e_b+lo_b, e_b+hi_b]
    相交，交集中的每个值都是一个可行的砍伐年（年表下标）。残失环不计入
    目标，三级目标仅由两芯映射本身决定。
    """
    ka, kb = _min_skips(fa), _min_skips(fb)
    lo_a, hi_a = loss_a
    lo_b, hi_b = loss_b
    pairs: list[tuple[int, int, int, int]] = []
    for ea in ka:
        for eb in kb:
            f_lo = max(ea + lo_a, eb + lo_b)
            f_hi = min(ea + hi_a, eb + hi_b)
            if f_lo <= f_hi:
                pairs.append((ea, eb, f_lo, f_hi))
    if not pairs:
        return None
    s_total = min(ka[ea] + kb[eb] for ea, eb, _, _ in pairs)
    pairs = [p for p in pairs if ka[p[0]] + kb[p[1]] == s_total]

    per_pair: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for ea, eb, f_lo, f_hi in pairs:
        pa = [r for r in fa[ea] if r.miss + r.fls == ka[ea]]
        pb = [r for r in fb[eb] if r.miss + r.fls == kb[eb]]
        m_e = max(min(r.mmax for r in pa), min(r.mmax for r in pb))
        # 联合最大差值恰为 m_e 时，两侧差值和各自可取的最小值。
        d_a = min(r.dsum for r in pa if r.mmax <= m_e)
        d_b = min(r.dsum for r in pb if r.mmax <= m_e)
        per_pair[(ea, eb, f_lo, f_hi)] = (m_e, d_a + d_b)
    mstar = min(v[0] for v in per_pair.values())
    sstar = min(v[1] for v in per_pair.values() if v[0] == mstar)
    best = sorted(p for p, v in per_pair.items() if v == (mstar, sstar))
    return {"skips": s_total, "mmax": mstar, "dsum": sstar,
            "pairs": best, "ka": ka, "kb": kb}


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


def _mapping_pairs(sa, sb, chron, ta, tb, ea, eb, skips_a, skips_b,
                   mstar, sstar, miss_caps, false_caps, limit):
    """固定两芯终年后，按映射序产出前 limit 份同优映射对 (ma, mb)。"""
    out: list[tuple[Mapping, Mapping]] = []
    b_cache: dict[int, list[Mapping]] = {}
    for ma in _enumerate_core(sa, chron, ta, ea, skips_a, mstar,
                              miss_caps[0], false_caps[0], None, None):
        need = sstar - ma.dsum
        if need not in b_cache:
            b_cache[need] = list(_enumerate_core(
                sb, chron, tb, eb, skips_b, mstar,
                miss_caps[1], false_caps[1], need, 2))
        for mb in b_cache[need]:
            out.append((ma, mb))
            if len(out) >= limit:
                return out
    return out


def find_witnesses_loss(sa, sb, chron, ta, tb,
                        miss_caps, false_caps, opt) -> list:
    """残失校准下跨所有最优终年对取前两份联合见证。

    每份见证为 (砍伐年下标, A 残失数, B 残失数, A 映射, B 映射)。
    同一终年对内砍伐年递增、同年按映射序；全局按
    （砍伐年, A 残失数, B 残失数, 映射序）归并取前两份，
    因此“唯一 / 歧义”判定精确覆盖全部同优解释。
    """
    mstar, sstar = opt["mmax"], opt["dsum"]
    pool: list[tuple[int, int, int, Mapping, Mapping]] = []
    for ea, eb, f_lo, f_hi in opt["pairs"]:
        mps = _mapping_pairs(sa, sb, chron, ta, tb, ea, eb,
                             opt["ka"][ea], opt["kb"][eb], mstar, sstar,
                             miss_caps, false_caps, 2)
        found: list[tuple[int, int, int, Mapping, Mapping]] = []
        for f in range(f_lo, f_hi + 1):
            for ma, mb in mps:
                found.append((f, f - ea, f - eb, ma, mb))
                if len(found) == 2:
                    break
            if len(found) == 2:
                break
        pool.extend(found)
    pool.sort(key=lambda t: (t[0], t[1], t[2], t[3].sig, t[4].sig))
    return pool[:2]


# --------------------------------------------------------------------------- #
# 响应组装
# --------------------------------------------------------------------------- #

def _mapping_payload(mp: Mapping, start_year: int, sample: list[int],
                     chron: list[int],
                     outer_loss: Optional[int] = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
    if outer_loss is not None:
        # 残失校准：末环匹配年之后的边材残失环数。
        payload["outer_loss"] = outer_loss
    return payload


def _witness_payload(ma: Mapping, mb: Mapping, start_year: int,
                     sa: list[int], sb: list[int],
                     chron: list[int],
                     felling: Optional[int] = None) -> dict[str, Any]:
    ea = ma.pairs[-1][1]
    eb = mb.pairs[-1][1]
    if felling is None:
        return {
            "common_end_year": start_year + ea,
            "mappings": {
                "a": _mapping_payload(ma, start_year, sa, chron),
                "b": _mapping_payload(mb, start_year, sb, chron),
            },
        }
    # 残失校准：common_end_year 为共同砍伐年，各芯保留实际末环年份。
    return {
        "common_end_year": start_year + felling,
        "mappings": {
            "a": _mapping_payload(ma, start_year, sa, chron,
                                  outer_loss=felling - ea),
            "b": _mapping_payload(mb, start_year, sb, chron,
                                  outer_loss=felling - eb),
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
                         caps, fa, fb, loss=None) -> dict[str, Any]:
    (miss_a, false_a), (miss_b, false_b) = caps
    n = (len(sa), len(sb))
    m = len(chron)

    rel8_a, k8_a = _relaxed_info(sa, chron, ta, m, n[0] - 8, 8)
    rel8_b, k8_b = _relaxed_info(sb, chron, tb, m, n[1] - 8, 8)
    rel1_a, k1_a = _relaxed_info(sa, chron, ta, m, n[0] - 1, 1)
    rel1_b, k1_b = _relaxed_info(sb, chron, tb, m, n[1] - 1, 1)

    last_a = [j for j in range(m) if abs(sa[-1] - chron[j]) <= ta[-1]]
    last_b = [j for j in range(m) if abs(sb[-1] - chron[j]) <= tb[-1]]

    if loss is None:
        shared_relaxed = sorted(set(rel1_a) & set(rel1_b))
    else:
        # 残失校准下，共同候选是“末环年 + 残失数”可达的砍伐年下标。
        shared_relaxed = sorted(
            {e + d for e in rel1_a
             for d in range(loss[0][0], loss[0][1] + 1)}
            & {e + d for e in rel1_b
               for d in range(loss[1][0], loss[1][1] + 1)}
        )

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

    def core_ev(rel8, rel1, k8, last, strict, miss_cap, false_cap, nn,
                loss_bounds=None):
        ev: dict[str, Any] = {
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
        if loss_bounds is not None:
            lo, hi = loss_bounds
            # 该芯在限额内可达末环年 + 残失区间推出的全部砍伐年。
            ev["reachable_felling_years_within_limits"] = sorted(
                {start_year + e + d for e in strict
                 for d in range(lo, hi + 1)})
            ev["limits"]["outer_loss"] = {"min": lo, "max": hi}
        return ev

    loss_a = loss[0] if loss is not None else None
    loss_b = loss[1] if loss is not None else None
    ev_a = core_ev(rel8_a, rel1_a, k8_a, last_a, fa, miss_a, false_a,
                   n[0], loss_a)
    ev_b = core_ev(rel8_b, rel1_b, k8_b, last_b, fb, miss_b, false_b,
                   n[1], loss_b)

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
            key = "ab"[idx]
            if loss is None:
                r = best8[idx].get(j) or best1[idx].get(j)
                rec: Optional[dict[str, Any]] = None
                if r is not None:
                    rec = {
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
                entry[key] = rec
                continue
            # 残失校准：该砍伐年可经多个 (末环, 残失数) 组合达到，
            # 取最优记录（优先 8 环匹配档，再按跳环 / 差值 / 残失数）。
            lo, hi = loss[idx]
            cands = []
            for d in range(lo, hi + 1):
                e = j - d
                for tier, pool in ((0, best8[idx]), (1, best1[idx])):
                    r = pool.get(e)
                    if r is not None:
                        cands.append((tier, r.miss + r.fls, r.mmax,
                                      r.dsum, d, e, r))
            if not cands:
                entry[key] = None
                continue
            cands.sort(key=lambda t: t[:6])
            _, _, _, _, d, e, r = cands[0]
            entry[key] = {
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
                "end_year": start_year + e,
                "outer_loss": d,
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

def align_joint(payload: dict[str, Any]) -> dict[str, Any]:
    chron: list[int] = payload["chronology"]
    start_year: int = payload["start_year"]
    sa, sb = payload["sample_a"], payload["sample_b"]
    ta, tb = payload["tolerance_a"], payload["tolerance_b"]
    miss_a, false_a = payload["max_missing_a"], payload["max_false_a"]
    miss_b, false_b = payload["max_missing_b"], payload["max_false_b"]
    caps = ((miss_a, false_a), (miss_b, false_b))

    tl = payload.get("terminal_loss")
    loss: Optional[tuple[tuple[int, int], tuple[int, int]]] = None
    if tl is not None:
        loss = ((tl["a"]["min"], tl["a"]["max"]),
                (tl["b"]["min"], tl["b"]["max"]))

    fa = core_front(sa, chron, ta, miss_a, false_a, 8)
    fb = core_front(sb, chron, tb, miss_b, false_b, 8)

    if loss is None:
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

    opt = _joint_objective_loss(fa, fb, loss[0], loss[1])
    if opt is None:
        return _no_solution_payload(start_year, chron, sa, sb, ta, tb,
                                    caps, fa, fb, loss=loss)
    witnesses = find_witnesses_loss(
        sa, sb, chron, ta, tb,
        (miss_a, miss_b), (false_a, false_b), opt,
    )
    if not witnesses:
        # 理论不可达（最优值必有映射支撑），保守按无解处理。
        return _no_solution_payload(start_year, chron, sa, sb, ta, tb,
                                    caps, fa, fb, loss=loss)
    f0, _, _, ma0, mb0 = witnesses[0]
    status = "unique" if len(witnesses) == 1 else "ambiguous"
    primary = _witness_payload(ma0, mb0, start_year, sa, sb, chron,
                               felling=f0)
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
            _witness_payload(ma, mb, start_year, sa, sb, chron, felling=f)
            for f, _, _, ma, mb in witnesses
        ],
        "evidence": None,
    }
