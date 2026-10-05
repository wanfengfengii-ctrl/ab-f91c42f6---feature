"""terminal_loss 残失校准的接口与算法测试，含小规模穷举对照。"""

from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient

from app.alignment import align_joint
from app.main import app
from test_alignment import _random_case, brute_joint, brute_paths

client = TestClient(app)

CHRON = list(range(100, 130))  # start_year=2000 时下标 j 即公元 2000+j 年


def _payload(sa, sb, loss_a, loss_b, ta=None, tb=None, chron=CHRON,
             start_year=2000, caps=(3, 3, 3, 3)):
    return dict(
        chronology=chron, start_year=start_year,
        sample_a=sa, sample_b=sb,
        tolerance_a=[0] * len(sa) if ta is None else ta,
        tolerance_b=[0] * len(sb) if tb is None else tb,
        max_missing_a=caps[0], max_false_a=caps[1],
        max_missing_b=caps[2], max_false_b=caps[3],
        terminal_loss={
            "a": {"min": loss_a[0], "max": loss_a[1]},
            "b": {"min": loss_b[0], "max": loss_b[1]},
        },
    )


# --------------------------------------------------------------------------- #
# 算法行为
# --------------------------------------------------------------------------- #

def test_loss_unique_felling_year():
    # A 末环 2015（下标 15），B 末环 2018（下标 18）；
    # A 残失 3 环后两芯同于 2018 年砍伐。
    p = _payload(CHRON[5:16], CHRON[8:19], (0, 3), (0, 0))
    res = align_joint(p)
    assert res["status"] == "unique"
    assert res["common_end_year"] == 2018
    assert res["objectives"] == {
        "total_skips": 0, "max_difference": 0, "difference_sum": 0}
    ma, mb = res["mappings"]["a"], res["mappings"]["b"]
    assert ma["end_year"] == 2015 and ma["outer_loss"] == 3
    assert mb["end_year"] == 2018 and mb["outer_loss"] == 0
    for key in ("a", "b"):
        mp = res["mappings"][key]
        assert mp["end_year"] + mp["outer_loss"] == res["common_end_year"]
    assert len(res["witnesses"]) == 1
    assert res["mappings"] == res["witnesses"][0]["mappings"]


def test_loss_not_counted_as_skips():
    # B 缺年表下标 17（1 个缺环），末环 2018；B 再残失 1 环至 2019。
    # 残失环不计入缺环 / 伪环 / 三级目标：总跳环仍为 1。
    chron = list(range(100, 140))
    sa = chron[9:20]                       # 末环 2019
    sb = [chron[i] for i in [8, 9, 10, 11, 12, 13, 14, 15, 16, 18]]
    p = _payload(sa, sb, (0, 0), (0, 2), chron=chron)
    res = align_joint(p)
    assert res["status"] == "unique"
    assert res["common_end_year"] == 2019
    assert res["objectives"] == {
        "total_skips": 1, "max_difference": 0, "difference_sum": 0}
    mb = res["mappings"]["b"]
    assert mb["missing_chronology_indices"] == [17]
    assert mb["missing_count"] == 1
    assert mb["end_year"] == 2018 and mb["outer_loss"] == 1
    assert res["mappings"]["a"]["outer_loss"] == 0


def test_loss_ambiguous_multiple_felling_years():
    # 同一对映射可配三种残失组合 -> 三个同优解释都参与唯一性判定，
    # 见证按（砍伐年, 两芯残失数, 映射序）取前两份。
    p = _payload(CHRON[5:16], CHRON[6:17], (0, 3), (0, 3))
    res = align_joint(p)
    assert res["status"] == "ambiguous"
    assert res["common_end_year"] == 2016
    ws = res["witnesses"]
    assert len(ws) == 2
    assert [w["common_end_year"] for w in ws] == [2016, 2017]
    assert (ws[0]["mappings"]["a"]["outer_loss"],
            ws[0]["mappings"]["b"]["outer_loss"]) == (1, 0)
    assert (ws[1]["mappings"]["a"]["outer_loss"],
            ws[1]["mappings"]["b"]["outer_loss"]) == (2, 1)
    # 两份见证的映射相同，仅残失解释不同
    def sig(w):
        return (
            tuple((m["sample_ring"], m["chronology_index"])
                  for m in w["mappings"]["a"]["matched"]),
            tuple((m["sample_ring"], m["chronology_index"])
                  for m in w["mappings"]["b"]["matched"]),
        )
    assert sig(ws[0]) == sig(ws[1])
    for w in ws:
        for key in ("a", "b"):
            mp = w["mappings"][key]
            assert mp["end_year"] + mp["outer_loss"] == w["common_end_year"]


def test_loss_no_common_felling_year():
    # A 末环 2013、B 末环 2029，残失区间各自 0–3 仍无法汇合，
    # 无解证据须列出双方可达的推定砍伐年。
    p = _payload(CHRON[3:14], CHRON[20:30], (0, 3), (0, 3))
    res = align_joint(p)
    assert res["status"] == "no_solution"
    assert res["common_end_year"] is None
    assert res["mappings"] is None
    ev = res["evidence"]
    assert ev["reason"] == "no_common_end_year"
    assert ev["shared_candidate_years"] == []
    assert ev["cores"]["a"]["reachable_end_years_within_limits"] == [2013]
    assert ev["cores"]["b"]["reachable_end_years_within_limits"] == [2029]
    assert ev["cores"]["a"]["reachable_felling_years_within_limits"] == [
        2013, 2014, 2015, 2016]
    assert ev["cores"]["b"]["reachable_felling_years_within_limits"] == [
        2029, 2030, 2031, 2032]
    assert ev["cores"]["a"]["limits"]["outer_loss"] == {"min": 0, "max": 3}
    assert ev["cores"]["b"]["limits"]["outer_loss"] == {"min": 0, "max": 3}


def test_loss_shared_end_details_when_relaxed_felling_shared():
    # A 末环值在下标 9 与 26 处同现：严格限额只能终 9（推定 2009–2012），
    # 放宽后可终 26（推定 2026–2029），与 B 的 2029 汇合。
    # 严格无法形成共同砍伐年 -> common_end_requires_more_skips，
    # shared_end_details 须给出各芯达到该砍伐年的末环年与残失数。
    chron = list(range(100, 130))
    chron[26] = 109
    sa = chron[0:9] + [chron[9]]
    sb = chron[20:30]
    p = _payload(sa, sb, (0, 3), (0, 3), chron=chron)
    res = align_joint(p)
    assert res["status"] == "no_solution"
    ev = res["evidence"]
    assert ev["reason"] == "common_end_requires_more_skips"
    assert ev["shared_candidate_years"] == [2029]
    (detail,) = ev["shared_end_details"]
    assert detail["year"] == 2029
    assert detail["a"]["end_year"] == 2026
    assert detail["a"]["outer_loss"] == 3
    assert detail["a"]["within_limits"] is False
    assert detail["b"]["end_year"] == 2029
    assert detail["b"]["outer_loss"] == 0
    assert detail["b"]["within_limits"] is True


def test_loss_zero_interval_matches_compat_result():
    # 两侧残失区间都为 [0,0] 时，砍伐年即末环年，与兼容调用结论一致。
    base = dict(
        chronology=CHRON, start_year=2000,
        sample_a=CHRON[5:16], sample_b=CHRON[8:19],
        tolerance_a=[0] * 11, tolerance_b=[0] * 11,
        max_missing_a=3, max_false_a=3, max_missing_b=3, max_false_b=3,
    )
    # 两芯末环不同年：兼容调用无解，残失 [0,3]/[0,0] 可解。
    res = align_joint(dict(base, terminal_loss={
        "a": {"min": 0, "max": 3}, "b": {"min": 0, "max": 0}}))
    assert res["status"] == "unique"
    assert res["common_end_year"] == 2018
    # 残失 [0,0]/[0,0] 与兼容调用同样无解。
    res00 = align_joint(dict(base, terminal_loss={
        "a": {"min": 0, "max": 0}, "b": {"min": 0, "max": 0}}))
    compat = align_joint(base)
    assert res00["status"] == compat["status"] == "no_solution"
    assert res00["evidence"]["reason"] == compat["evidence"]["reason"]


# --------------------------------------------------------------------------- #
# 接口校验
# --------------------------------------------------------------------------- #

def _api_req():
    chron = [100 + (i * 7) % 50 for i in range(30)]
    sa = [chron[i] for i in range(9, 20)]
    sb = [chron[i] for i in [9, 10, 12, 13, 14, 15, 16, 17, 18, 19]]
    return dict(
        chronology=chron, start_year=1990,
        sample_a=sa, sample_b=sb,
        tolerance_a=[0] * 11, tolerance_b=[0] * 10,
        max_missing_a=3, max_false_a=3, max_missing_b=3, max_false_b=3,
    )


def _locs(body):
    return [tuple(e.get("loc", ())) for e in body["errors"]]


def test_api_terminal_loss_ok():
    p = _api_req()
    p["terminal_loss"] = {"a": {"min": 0, "max": 0}, "b": {"min": 0, "max": 0}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "unique"
    assert body["common_end_year"] == 2009
    assert body["mappings"]["a"]["outer_loss"] == 0
    assert body["mappings"]["b"]["outer_loss"] == 0
    assert body["mappings"]["a"]["end_year"] == 2009


def test_api_terminal_loss_missing_side():
    p = _api_req()
    p["terminal_loss"] = {"a": {"min": 0, "max": 2}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "validation_error"
    assert any("terminal_loss" in loc and "b" in loc for loc in _locs(body))


def test_api_terminal_loss_missing_bound():
    p = _api_req()
    p["terminal_loss"] = {"a": {"max": 2}, "b": {"min": 0, "max": 1}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422
    locs = _locs(r.json())
    assert any("terminal_loss" in loc and "a" in loc and "min" in loc
               for loc in locs)


def test_api_terminal_loss_inverted_interval():
    p = _api_req()
    p["terminal_loss"] = {"a": {"min": 3, "max": 1},
                          "b": {"min": 0, "max": 1}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422
    body = r.json()
    assert any("terminal_loss" in loc and "a" in loc
               for loc in _locs(body))
    assert any("倒置" in str(e.get("msg", "")) for e in body["errors"])


def test_api_terminal_loss_out_of_range():
    p = _api_req()
    p["terminal_loss"] = {"a": {"min": 0, "max": 4},
                          "b": {"min": 0, "max": 1}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422
    p = _api_req()
    p["terminal_loss"] = {"a": {"min": -1, "max": 2},
                          "b": {"min": 0, "max": 1}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422


def test_api_terminal_loss_extra_key_rejected():
    p = _api_req()
    p["terminal_loss"] = {"a": {"min": 0, "max": 1, "note": "x"},
                          "b": {"min": 0, "max": 1}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422


def test_api_terminal_loss_non_integer_rejected():
    p = _api_req()
    p["terminal_loss"] = {"a": {"min": 0.5, "max": 1},
                          "b": {"min": 0, "max": 1}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# 兼容性：未提供 terminal_loss 时响应结构不变
# --------------------------------------------------------------------------- #

def test_compat_response_has_no_outer_loss():
    r = client.post("/api/dendro/align", json=_api_req())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "unique"
    assert "outer_loss" not in body["mappings"]["a"]
    assert "outer_loss" not in body["mappings"]["b"]
    for w in body["witnesses"]:
        assert "outer_loss" not in w["mappings"]["a"]
        assert "outer_loss" not in w["mappings"]["b"]


def test_compat_no_solution_evidence_unchanged():
    p = _api_req()
    p["sample_b"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 99999]
    p["tolerance_b"] = [0] * 10
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no_solution"
    cores = body["evidence"]["cores"]
    assert "reachable_felling_years_within_limits" not in cores["a"]
    assert "reachable_felling_years_within_limits" not in cores["b"]
    assert "outer_loss" not in cores["a"]["limits"]
    assert "outer_loss" not in cores["b"]["limits"]


# --------------------------------------------------------------------------- #
# 回归：最后间隙（末两对匹配之间）含缺环 / 伪环的映射必须可枚举
# --------------------------------------------------------------------------- #

def test_regression_missing_ring_in_final_gap():
    # 缺环紧邻末环前（年表下标 19）：阶段 A 本就算出最优，
    # 阶段 B 也必须能枚举出该映射，而不是保守无解。
    chron = list(range(100, 140))
    s = [chron[i] for i in [10, 11, 12, 13, 14, 15, 16, 17, 18, 20]]
    p = dict(chronology=chron, start_year=2000, sample_a=s, sample_b=s,
             tolerance_a=[0] * 10, tolerance_b=[0] * 10,
             max_missing_a=3, max_false_a=3, max_missing_b=3, max_false_b=3)
    res = align_joint(p)
    assert res["status"] == "unique"
    assert res["common_end_year"] == 2020
    assert res["objectives"] == {
        "total_skips": 2, "max_difference": 0, "difference_sum": 0}
    assert res["mappings"]["a"]["missing_chronology_indices"] == [19]
    assert res["mappings"]["b"]["missing_chronology_indices"] == [19]


def test_regression_false_ring_in_final_gap():
    # 伪环位于倒数第二位（样芯第 10 环）。
    chron = list(range(100, 140))
    s = [chron[i] for i in range(10, 19)] + [999] + [chron[19]]
    p = dict(chronology=chron, start_year=2000, sample_a=s, sample_b=s,
             tolerance_a=[0] * 11, tolerance_b=[0] * 11,
             max_missing_a=3, max_false_a=3, max_missing_b=3, max_false_b=3)
    res = align_joint(p)
    assert res["status"] == "unique"
    assert res["common_end_year"] == 2019
    assert res["objectives"]["total_skips"] == 2
    assert res["mappings"]["a"]["false_sample_rings"] == [10]
    assert res["mappings"]["b"]["false_sample_rings"] == [10]


def _random_tail_case(rng, m=26):
    """生成最后间隙含缺环 / 伪环的随机算例。"""
    chron = [rng.randint(0, 9) for _ in range(m)]

    def make_core():
        start = rng.randint(0, m - 13)
        idx = list(range(start, start + 10))
        # 中间缺环（保持末两位之外的位置）
        if rng.random() < 0.5:
            k = rng.randint(1, len(idx) - 3)
            idx = idx[:k] + [x + 1 for x in idx[k:]]
        # 最后间隙缺环：拉开末两对的年表下标
        idx[-1] += rng.randint(0, 2)
        # 精确值 + 全零容差：映射被真值锁死，尾部跳环必进入最优解
        vals = [chron[j] for j in idx]
        # 最后间隙伪环：倒数第二位插入随机值
        if rng.random() < 0.5:
            vals.insert(len(vals) - 1, rng.randint(0, 9))
        tol = [0] * len(vals)
        return vals, tol

    sa, ta = make_core()
    sb, tb = make_core()
    return chron, sa, sb, ta, tb


@pytest.mark.parametrize("seed", range(40))
def test_random_tail_gap_cases_match_bruteforce(seed):
    rng = random.Random(9000 + seed)
    chron, sa, sb, ta, tb = _random_tail_case(rng)
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
    expect_n = 1 if len(opt_pairs) == 1 else 2
    assert len(res["witnesses"]) == expect_n
    first = opt_pairs[0]
    got_a = tuple((m["sample_ring"] - 1, m["chronology_index"])
                  for m in res["witnesses"][0]["mappings"]["a"]["matched"])
    got_b = tuple((m["sample_ring"] - 1, m["chronology_index"])
                  for m in res["witnesses"][0]["mappings"]["b"]["matched"])
    assert (got_a, got_b) == (first[0].sig, first[1].sig)


# --------------------------------------------------------------------------- #
# 穷举对照（随机小规模，含残失区间）
# --------------------------------------------------------------------------- #

def brute_joint_loss(sa, sb, chron, ta, tb, caps, loss_a, loss_b):
    """穷举残失校准下的全部同优解释：(score, 砍伐年, A残失, B残失, A映射, B映射)。"""
    pa = brute_paths(sa, chron, ta, caps[0], caps[1])
    pb = brute_paths(sb, chron, tb, caps[2], caps[3])
    best = None
    interps = []
    for ea, la in pa.items():
        for eb, lb in pb.items():
            f_lo = max(ea + loss_a[0], eb + loss_b[0])
            f_hi = min(ea + loss_a[1], eb + loss_b[1])
            if f_lo > f_hi:
                continue
            for a in la:
                for b in lb:
                    score = (a.skips + b.skips, max(a.mmax, b.mmax),
                             a.dsum + b.dsum)
                    if best is None or score < best:
                        best = score
                    for f in range(f_lo, f_hi + 1):
                        interps.append((score, f, f - ea, f - eb, a, b))
    if best is None:
        return None, []
    interps = [t for t in interps if t[0] == best]
    interps.sort(key=lambda t: (t[1], t[2], t[3], t[4].sig, t[5].sig))
    return best, interps


@pytest.mark.parametrize("seed", range(40))
def test_random_loss_cases_match_bruteforce(seed):
    rng = random.Random(5000 + seed)
    chron, sa, sb, ta, tb = _random_case(rng)
    if len(sa) < 10 or len(sb) < 10:
        return
    lo_a = rng.randint(0, 3)
    hi_a = rng.randint(lo_a, 3)
    lo_b = rng.randint(0, 3)
    hi_b = rng.randint(lo_b, 3)
    p = dict(
        chronology=chron, start_year=1900,
        sample_a=sa, sample_b=sb, tolerance_a=ta, tolerance_b=tb,
        max_missing_a=3, max_false_a=3, max_missing_b=3, max_false_b=3,
        terminal_loss={"a": {"min": lo_a, "max": hi_a},
                       "b": {"min": lo_b, "max": hi_b}},
    )
    best, interps = brute_joint_loss(
        sa, sb, chron, ta, tb, (3, 3, 3, 3), (lo_a, hi_a), (lo_b, hi_b))
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
    expect_n = 1 if len(interps) == 1 else 2
    assert len(res["witnesses"]) == expect_n
    assert res["status"] == ("unique" if expect_n == 1 else "ambiguous")
    for k in range(expect_n):
        _, f, la, lb, ma, mb = interps[k]
        w = res["witnesses"][k]
        assert w["common_end_year"] == 1900 + f
        assert w["mappings"]["a"]["outer_loss"] == la
        assert w["mappings"]["b"]["outer_loss"] == lb
        assert w["mappings"]["a"]["end_year"] == 1900 + f - la
        assert w["mappings"]["b"]["end_year"] == 1900 + f - lb
        got_a = tuple((m["sample_ring"] - 1, m["chronology_index"])
                      for m in w["mappings"]["a"]["matched"])
        got_b = tuple((m["sample_ring"] - 1, m["chronology_index"])
                      for m in w["mappings"]["b"]["matched"])
        assert (got_a, got_b) == (ma.sig, mb.sig)
    assert res["common_end_year"] == res["witnesses"][0]["common_end_year"]
    assert res["mappings"] == res["witnesses"][0]["mappings"]
