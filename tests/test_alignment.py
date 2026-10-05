"""两芯联合对齐算法测试，含小规模穷举对照。"""

from __future__ import annotations

import itertools
import random

import pytest

from app.alignment import align_joint, Mapping


# --------------------------------------------------------------------------- #
# 穷举参考实现
# --------------------------------------------------------------------------- #

def brute_paths(sample, chron, tol, miss_cap, false_cap, min_matches=8):
    """穷举单芯所有合法映射：{end: [Mapping]}。

    映射即严格双增匹配对 (i_k, j_k)：首对 i_0=0、末对 i=n-1；
    相邻对之间未匹配的样芯环为伪环、年表项为缺环。
    """
    n, m = len(sample), len(chron)
    out: dict[int, list[Mapping]] = {}

    def rec(i, j_lo, pairs, miss_used, false_used):
        # i：当前待决策的样芯下标；j_lo：年表下一对的最小下标。
        if i >= n:
            return
        # 分支 1：把第 i 环当伪环跳过（首环与末环不允许）。
        if 0 < i < n - 1 and false_used < false_cap:
            rec(i + 1, j_lo, pairs, miss_used, false_used + 1)
        # 分支 2：第 i 环匹配年表第 j 项。
        prev_j = pairs[-1][1] if pairs else None
        for j in range(j_lo, m):
            add_miss = 0 if prev_j is None else j - prev_j - 1
            if miss_used + add_miss > miss_cap:
                break
            d = abs(sample[i] - chron[j])
            if d > tol[i]:
                continue
            p = pairs + [(i, j, d)]
            if i == n - 1:
                matched_i = {pp[0] for pp in p}
                matched_j = {pp[1] for pp in p}
                j0 = p[0][1]
                all_false = tuple(
                    q for q in range(n) if q not in matched_i)
                all_miss = tuple(
                    q for q in range(j0, j + 1) if q not in matched_j)
                mp = Mapping(tuple(p), all_miss, all_false)
                if (mp.miss <= miss_cap and mp.fls <= false_cap
                        and len(p) >= min_matches):
                    out.setdefault(j, []).append(mp)
            else:
                rec(i + 1, j + 1, p, miss_used + add_miss, false_used)

    rec(0, 0, [], 0, 0)
    return out


def brute_joint(sa, sb, chron, ta, tb, ma_c, fa_c, mb_c, fb_c):
    pa = brute_paths(sa, chron, ta, ma_c, fa_c)
    pb = brute_paths(sb, chron, tb, mb_c, fb_c)
    best = None
    pairs_opt = []
    for e in sorted(set(pa) & set(pb)):
        for a, b in itertools.product(pa[e], pb[e]):
            score = (a.skips + b.skips, max(a.mmax, b.mmax),
                     a.dsum + b.dsum)
            if best is None or score < best:
                best = score
                pairs_opt = [(a, b)]
            elif score == best:
                pairs_opt.append((a, b))
    pairs_opt.sort(key=lambda t: (t[0].sig, t[1].sig))
    return best, pairs_opt


def make_payload(chron, sa, sb, ta=None, tb=None, tol=2, start_year=2000,
                 caps=(3, 3, 3, 3)):
    return dict(
        chronology=chron, start_year=start_year,
        sample_a=sa, sample_b=sb,
        tolerance_a=[tol] * len(sa) if ta is None else ta,
        tolerance_b=[tol] * len(sb) if tb is None else tb,
        max_missing_a=caps[0], max_false_a=caps[1],
        max_missing_b=caps[2], max_false_b=caps[3],
    )


# --------------------------------------------------------------------------- #
# 构造场景
# --------------------------------------------------------------------------- #

def test_basic_unique_alignment():
    chron = [100 + (i * 7) % 50 for i in range(40)]
    sa = [chron[i] for i in range(5, 17)]       # 12 环，终年 16
    sb = [chron[i] for i in [8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    # A 终年调整为 20 与 B 一致
    sa = [chron[i] for i in range(9, 21)]       # 12 环，终年 20
    # B 在 10 处缺环
    assert sb[1] != sb[2]
    res = align_joint(make_payload(chron, sa, sb, ta=[0] * 12, tb=[0] * 12))
    assert res["status"] == "unique"
    assert res["common_end_year"] == 2020
    assert res["objectives"] == {
        "total_skips": 1, "max_difference": 0, "difference_sum": 0}
    wa = res["mappings"]["a"]
    wb = res["mappings"]["b"]
    assert wa["missing_chronology_indices"] == []
    assert wa["false_sample_rings"] == []
    assert wb["missing_chronology_indices"] == [10]
    assert wb["false_sample_rings"] == []
    assert wa["match_count"] == 12
    assert wb["match_count"] == 12
    assert len(res["witnesses"]) == 1


def test_false_ring_skipped_in_sample():
    chron = [50 + i * 3 for i in range(40)]
    # A 芯 11 环：第 5 环是伪环（不在年表），其余取年表 3..13，终年 13
    sa = [chron[3], chron[4], chron[5], chron[6], chron[7],
          999,
          chron[8], chron[9], chron[10], chron[11], chron[12 + 1]]
    # 重新设计：年表 3..12 共 10 年，插入 1 个伪环 -> 11 环，终年 12
    sa = [chron[3], chron[4], chron[5], chron[6], chron[7], 999,
          chron[8], chron[9], chron[10], chron[11], chron[12]]
    sb = [chron[i] for i in range(2, 13)]  # 11 环，终年 12
    res = align_joint(make_payload(chron, sa, sb))
    assert res["status"] == "unique"
    assert res["common_end_year"] == 2012
    assert res["objectives"]["total_skips"] == 1
    assert res["mappings"]["a"]["false_sample_rings"] == [6]
    assert res["mappings"]["a"]["missing_chronology_indices"] == []
    assert res["mappings"]["b"]["false_sample_rings"] == []


def test_missing_and_false_caps_zero():
    chron = [100 + (i * 11) % 47 for i in range(35)]
    # 两芯终年同为 18；B 缺年表下标 9。
    sa = [chron[i] for i in range(8, 19)]       # 11 环
    sb = [chron[i] for i in [8, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    p = make_payload(chron, sa, sb, ta=[0] * 11, tb=[0] * 10,
                     caps=(3, 3, 0, 3))
    res = align_joint(p)
    assert res["status"] == "no_solution"
    assert res["evidence"]["cores"]["b"]["limits"]["missing"] == 0
    # 放开缺环限额后可解
    p["max_missing_b"] = 1
    res = align_joint(p)
    assert res["status"] == "unique"
    assert res["common_end_year"] == 2018
    assert res["mappings"]["b"]["missing_chronology_indices"] == [9]


def test_no_common_end_year():
    # A 精确落在下标 3..14（终年 14，最多再插 3 个缺环 -> 最晚终年 17）；
    # B 精确落在下标 16..25（终年 25）。两者终年集合不相交。
    chron = list(range(100, 130))
    sa = chron[3:15]
    sb = chron[16:26]
    p = make_payload(chron, sa, sb, ta=[0] * 12, tb=[0] * 10)
    res = align_joint(p)
    assert res["status"] == "no_solution"
    assert res["evidence"]["reason"] == "no_common_end_year"
    assert res["evidence"]["shared_candidate_years"] == []
    assert res["evidence"]["cores"]["a"][
        "reachable_end_years_within_limits"]
    assert res["evidence"]["cores"]["b"][
        "reachable_end_years_within_limits"] == [2025]


def test_last_ring_unmatchable_evidence():
    chron = list(range(100, 130))
    sa = chron[3:15]
    sb = [101, 102, 103, 104, 105, 106, 107, 108, 109, 999]
    res = align_joint(make_payload(chron, sa, sb, ta=[0] * 12, tb=[0] * 10))
    assert res["status"] == "no_solution"
    assert res["evidence"]["reason"] == "last_ring_unmatchable"
    assert res["evidence"]["cores"]["b"]["last_ring_candidate_years"] == []


def test_skip_limits_exceeded_evidence():
    # 两芯真值都需要 4 个缺环，超过各自限额 3。
    chron = list(range(100, 160))
    sa = [chron[i] for i in list(range(0, 6)) + list(range(10, 14))]
    sb = sa[:]
    res = align_joint(make_payload(chron, sa, sb, ta=[0] * 10, tb=[0] * 10))
    assert res["status"] == "no_solution"
    assert res["evidence"]["reason"] == "skip_limits_exceeded"
    ev = res["evidence"]
    assert ev["cores"]["a"]["min_skips_within_limits"] is None
    # 放宽限额（缺环允许 4 无法通过 API，此处直接核对放宽后信息存在）
    assert ev["cores"]["a"]["min_skips_with_eight_matches"] is not None
    assert "shared_end_details" in ev


def test_insufficient_matchable_rings_evidence():
    # A 芯有 3 个中环值在年表中根本不存在（容差 0），即便全部按伪环跳掉，
    # 伪环限额 3 时仍只剩 7 环可匹配，不足 8 环。
    chron = list(range(100, 130))
    sa = [chron[0], -1, -2, -3, chron[4], chron[5], chron[6],
          chron[7], chron[8], chron[9]]
    sb = chron[0:10]
    res = align_joint(make_payload(chron, sa, sb, ta=[0] * 10, tb=[0] * 10))
    assert res["status"] == "no_solution"
    assert res["evidence"]["reason"] == "insufficient_matchable_rings"
    assert res["evidence"]["cores"]["a"]["min_skips_with_eight_matches"] is None
    assert res["evidence"]["cores"]["a"]["min_skips_with_any_match"] is not None


def test_ambiguity_two_witnesses_sorted_by_mapping_order():
    # 构造两支都存在两种同优落点的情形：年表周期性重复。
    chron = [10] * 25 + [20] * 10
    sa = [10] * 12
    sb = [10] * 10
    p = make_payload(chron, sa, sb, ta=[0] * 12, tb=[0] * 10)
    res = align_joint(p)
    assert res["status"] == "ambiguous"
    assert len(res["witnesses"]) == 2
    # 见证按映射序：第一份终年早于第二份（j0=0 起，最早终年 11）。
    ends = [w["common_end_year"] for w in res["witnesses"]]
    assert ends[0] <= ends[1]
    assert res["common_end_year"] == ends[0]
    sigs = [
        tuple((m["sample_ring"], m["chronology_index"])
              for m in w["mappings"]["a"]["matched"])
        for w in res["witnesses"]
    ]
    assert sigs[0] <= sigs[1]


def test_tiebreak_max_diff_before_sum():
    # 终年同为 26、A 芯同为 4 跳环（缺 2 + 伪 2，8 环匹配）的候选：
    #   胜者 mmax=2、dsum=7（缺年表 22,23，伪环为样芯第 2、4 环）；
    #   败者 mmax=3、dsum=6（差值和更小，但最大差值更大）。
    # 次级目标先压最大差值，故胜者入选，尽管其差值和更大。
    chron = [1, 6, 6, 4, 8, 7, 6, 9, 6, 2, 0, 7, 9, 7, 2, 2, 4,
             7, 1, 2, 9, 7, 0, 2, 4, 7, 4, 5]
    sa = [8, 9, 1, 9, 0, 9, 5, 4, 8, 5]
    ta = [1, 2, 1, 1, 3, 1, 3, 1, 1, 3]
    sb = chron[17:27]                     # 精确落在年表下标 17..26
    tb = [0] * 10
    res = align_joint(make_payload(chron, sa, sb, ta=ta, tb=tb, start_year=2000))
    assert res["common_end_year"] == 2026
    assert res["objectives"]["total_skips"] == 4
    assert res["objectives"]["max_difference"] == 2
    assert res["objectives"]["difference_sum"] == 7
    ma = res["mappings"]["a"]
    assert ma["missing_chronology_indices"] == [22, 23]
    assert ma["false_sample_rings"] == [2, 4]
    assert ma["match_count"] == 8
    assert res["mappings"]["b"]["max_difference"] == 0
    assert res["mappings"]["b"]["match_count"] == 10


# --------------------------------------------------------------------------- #
# 穷举对照（随机小规模）
# --------------------------------------------------------------------------- #

def _random_case(rng, m=26, na=10, nb=10):
    chron = [rng.randint(0, 9) for _ in range(m)]
    # 以年表子序列为真值，注入噪声与缺/伪环
    def make_core():
        n = na
        start = rng.randint(0, m - n - 3)
        idx = list(range(start, start + n))
        # 随机把 0-2 个位置变缺环（抽掉年表项，顺延取后续年）
        miss = rng.randint(0, 2)
        for _ in range(miss):
            if idx[-1] + 1 < m:
                k = rng.randint(1, len(idx) - 2)
                idx = idx[:k] + [x + 1 for x in idx[k:]]
        vals = [chron[j] + rng.choice([-1, 0, 1]) for j in idx]
        # 随机插伪环（同时补逐环容差）
        false = rng.randint(0, 2)
        for _ in range(false):
            k = rng.randint(1, len(vals) - 1)
            vals.insert(k, rng.randint(0, 9))
        tol = [rng.randint(0, 2) for _ in vals]
        return vals, tol
    sa, ta = make_core()
    sb, tb = make_core()
    return chron, sa, sb, ta, tb


@pytest.mark.parametrize("seed", range(60))
def test_random_cases_match_bruteforce(seed):
    rng = random.Random(seed)
    chron, sa, sb, ta, tb = _random_case(rng)
    if len(sa) < 10 or len(sb) < 10:
        return
    p = dict(
        chronology=chron, start_year=1900,
        sample_a=sa, sample_b=sb, tolerance_a=ta, tolerance_b=tb,
        max_missing_a=3, max_false_a=3, max_missing_b=3, max_false_b=3,
    )
    best, opt_pairs = brute_joint(sa, sb, chron, ta, tb, 3, 3, 3, 3)
    res = align_joint(p)
    if best is None:
        assert res["status"] == "no_solution"
        return
    assert res["status"] in ("unique", "ambiguous")
    assert res["objectives"] == {
        "total_skips": best[0],
        "max_difference": best[1],
        "difference_sum": best[2],
    }
    assert res["common_end_year"] - 1900 == opt_pairs[0][0].pairs[-1][1]
    # 见证数量与映射序前两份
    expect_n = 1 if len(opt_pairs) == 1 else 2
    assert len(res["witnesses"]) == expect_n
    assert res["status"] == ("unique" if expect_n == 1 else "ambiguous")
    first = opt_pairs[0]
    got_a = tuple((m["sample_ring"] - 1, m["chronology_index"])
                  for m in res["witnesses"][0]["mappings"]["a"]["matched"])
    assert got_a == first[0].sig
    got_b = tuple((m["sample_ring"] - 1, m["chronology_index"])
                  for m in res["witnesses"][0]["mappings"]["b"]["matched"])
    assert got_b == first[1].sig
    if expect_n == 2:
        second = opt_pairs[1]
        got_a2 = tuple((m["sample_ring"] - 1, m["chronology_index"])
                       for m in res["witnesses"][1]["mappings"]["a"]["matched"])
        got_b2 = tuple((m["sample_ring"] - 1, m["chronology_index"])
                       for m in res["witnesses"][1]["mappings"]["b"]["matched"])
        assert (got_a2, got_b2) == (second[0].sig, second[1].sig)
    # 主映射即第一见证
    assert res["mappings"] == res["witnesses"][0]["mappings"]


def test_witnesses_payload_internal_consistency():
    chron = [100 + (i * 7) % 50 for i in range(32)]
    sa = [chron[i] + ((i * 3) % 2) for i in range(4, 18)]   # 14 环
    sb = [chron[i] for i in [6, 7, 9, 10, 11, 13, 14, 15, 16, 17]]
    res = align_joint(dict(
        chronology=chron, start_year=1850, sample_a=sa, sample_b=sb,
        tolerance_a=[2] * 14, tolerance_b=[1] * 10,
        max_missing_a=3, max_false_a=3, max_missing_b=3, max_false_b=3))
    assert res["status"] in ("unique", "ambiguous")
    for w in res["witnesses"]:
        for key, sample, tol in (("a", sa, [2] * 14), ("b", sb, [1] * 10)):
            mp = w["mappings"][key]
            assert mp["match_count"] >= 8
            assert mp["end_year"] == w["common_end_year"]
            years = [m["year"] for m in mp["matched"]]
            assert years == sorted(set(years))
            rings = [m["sample_ring"] for m in mp["matched"]]
            assert rings == sorted(set(rings))
            for m in mp["matched"]:
                assert m["difference"] == abs(
                    m["sample_value"] - m["chronology_value"])
                assert m["difference"] <= tol[m["sample_ring"] - 1]
            assert mp["missing_count"] == len(
                mp["missing_chronology_indices"])
            assert mp["false_count"] == len(mp["false_sample_rings"])
            assert mp["match_count"] == len(sample) - mp["false_count"]
            assert mp["max_difference"] == max(
                m["difference"] for m in mp["matched"])
            assert mp["difference_sum"] == sum(
                m["difference"] for m in mp["matched"])
        # 两芯终年一致
        assert w["mappings"]["a"]["end_year"] == w["mappings"]["b"]["end_year"]
