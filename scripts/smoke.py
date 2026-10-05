#!/usr/bin/env python3
"""两芯联合对齐 API 冒烟脚本。

兼容调用（未提供 terminal_loss）覆盖四类响应：唯一解、歧义解、
越界校验错误、无解（含两芯证据）；残失校准（terminal_loss）覆盖：
共同砍伐年唯一解、无法汇合时的双方可达推定年份、区间校验错误。
任一断言失败以非零退出码结束。
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def post(base: str, path: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


CHRON = [100 + (i * 7) % 50 for i in range(30)]


def case_unique() -> dict:
    sa = [CHRON[i] for i in range(9, 20)]
    sb = [CHRON[i] for i in [9, 10, 12, 13, 14, 15, 16, 17, 18, 19]]
    return dict(
        chronology=CHRON, start_year=1990,
        sample_a=sa, sample_b=sb,
        tolerance_a=[0] * 11, tolerance_b=[0] * 10,
        max_missing_a=3, max_false_a=3, max_missing_b=3, max_false_b=3,
    )


def case_ambiguous() -> dict:
    chron = [10] * 25 + [20] * 10
    return dict(
        chronology=chron, start_year=1990,
        sample_a=[10] * 12, sample_b=[10] * 10,
        tolerance_a=[0] * 12, tolerance_b=[0] * 10,
        max_missing_a=3, max_false_a=3, max_missing_b=3, max_false_b=3,
    )


def case_no_solution() -> dict:
    body = case_unique()
    body["sample_b"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 99999]
    body["tolerance_b"] = [0] * 10
    return body


def case_terminal_loss() -> dict:
    # A 末环 2015、B 末环 2018；A 残失 3 环后两芯同于 2018 年砍伐。
    chron = list(range(100, 130))
    return dict(
        chronology=chron, start_year=2000,
        sample_a=chron[5:16], sample_b=chron[8:19],
        tolerance_a=[0] * 11, tolerance_b=[0] * 11,
        max_missing_a=3, max_false_a=3, max_missing_b=3, max_false_b=3,
        terminal_loss={"a": {"min": 0, "max": 3},
                       "b": {"min": 0, "max": 0}},
    )


def case_terminal_loss_no_solution() -> dict:
    # A 末环 2013、B 末环 2029，残失区间各自 0–3 仍无法汇合。
    chron = list(range(100, 130))
    return dict(
        chronology=chron, start_year=2000,
        sample_a=chron[3:14], sample_b=chron[20:30],
        tolerance_a=[0] * 11, tolerance_b=[0] * 10,
        max_missing_a=3, max_false_a=3, max_missing_b=3, max_false_b=3,
        terminal_loss={"a": {"min": 0, "max": 3},
                       "b": {"min": 0, "max": 3}},
    )


def main(base: str) -> int:
    # 1) 唯一解
    code, r = post(base, "/api/dendro/align", case_unique())
    assert code == 200, r
    assert r["status"] == "unique", r
    assert r["common_end_year"] == 2009, r
    assert r["objectives"]["total_skips"] == 1, r
    assert r["mappings"]["a"]["end_year"] == r["mappings"]["b"]["end_year"]
    assert len(r["witnesses"]) == 1
    print(f"  [ok] 唯一解：终年 {r['common_end_year']}，"
          f"跳环 {r['objectives']['total_skips']}，"
          f"最大差 {r['objectives']['max_difference']}，"
          f"差值和 {r['objectives']['difference_sum']}")

    # 2) 歧义解：两份同优见证
    code, r = post(base, "/api/dendro/align", case_ambiguous())
    assert code == 200, r
    assert r["status"] == "ambiguous", r
    assert len(r["witnesses"]) == 2, r
    sigs = [
        tuple((m["sample_ring"], m["chronology_index"])
              for m in w["mappings"]["a"]["matched"])
        for w in r["witnesses"]
    ]
    assert sigs[0] < sigs[1], "见证须按映射序稳定排列"
    for w in r["witnesses"]:
        assert w["mappings"]["a"]["end_year"] == w["mappings"]["b"]["end_year"]
    print(f"  [ok] 歧义解：终年 {r['common_end_year']}，两份同优见证")

    # 3) 越界校验错误
    bad = case_unique()
    bad["chronology"] = [1] * 24
    code, r = post(base, "/api/dendro/align", bad)
    assert code == 422, r
    assert r["status"] == "validation_error", r
    assert r["errors"], r
    print("  [ok] 越界输入返回 422 校验错误")

    bad = case_unique()
    bad["max_missing_a"] = 4
    code, r = post(base, "/api/dendro/align", bad)
    assert code == 422, r
    print("  [ok] 缺环限额越界返回 422")

    # 4) 无解 + 两芯证据
    code, r = post(base, "/api/dendro/align", case_no_solution())
    assert code == 200, r
    assert r["status"] == "no_solution", r
    assert r["common_end_year"] is None
    ev = r["evidence"]
    assert ev["reason"] == "last_ring_unmatchable", ev
    assert set(ev["cores"]) == {"a", "b"}, ev
    assert ev["cores"]["b"]["last_ring_candidate_years"] == []
    assert "shared_end_details" in ev
    print("  [ok] 无解：原因 "
          f"{ev['reason']}，并给出两支树芯证据")

    # 5) 残失校准：末环不同年，残失后共同砍伐年
    code, r = post(base, "/api/dendro/align", case_terminal_loss())
    assert code == 200, r
    assert r["status"] == "unique", r
    assert r["common_end_year"] == 2018, r
    assert r["objectives"]["total_skips"] == 0, r
    ma, mb = r["mappings"]["a"], r["mappings"]["b"]
    assert ma["end_year"] == 2015 and ma["outer_loss"] == 3, r
    assert mb["end_year"] == 2018 and mb["outer_loss"] == 0, r
    for w in r["witnesses"]:
        for key in ("a", "b"):
            mp = w["mappings"][key]
            assert mp["end_year"] + mp["outer_loss"] == w["common_end_year"]
    print(f"  [ok] 残失校准：共同砍伐年 {r['common_end_year']}，"
          f"A 残失 {ma['outer_loss']} 环、B 残失 {mb['outer_loss']} 环")

    # 6) 残失校准无解：列出双方可达的推定砍伐年
    code, r = post(base, "/api/dendro/align", case_terminal_loss_no_solution())
    assert code == 200, r
    assert r["status"] == "no_solution", r
    assert r["common_end_year"] is None
    ev = r["evidence"]
    assert ev["reason"] == "no_common_end_year", ev
    assert ev["cores"]["a"]["reachable_felling_years_within_limits"] == [
        2013, 2014, 2015, 2016], ev
    assert ev["cores"]["b"]["reachable_felling_years_within_limits"] == [
        2029, 2030, 2031, 2032], ev
    print("  [ok] 残失校准无解：双方可达推定年份已列出")

    # 7) 残失配置校验错误
    bad = case_terminal_loss()
    bad["terminal_loss"] = {"a": {"min": 3, "max": 1},
                            "b": {"min": 0, "max": 1}}
    code, r = post(base, "/api/dendro/align", bad)
    assert code == 422, r
    assert r["status"] == "validation_error", r
    print("  [ok] 残失区间倒置返回 422 校验错误")

    bad = case_terminal_loss()
    del bad["terminal_loss"]["b"]
    code, r = post(base, "/api/dendro/align", bad)
    assert code == 422, r
    assert r["status"] == "validation_error", r
    print("  [ok] 残失配置缺侧返回 422 校验错误")

    print("\n冒烟全部通过。")
    return 0


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "http://app:8000"
    try:
        sys.exit(main(base.rstrip("/")))
    except AssertionError as e:
        print("冒烟断言失败：", e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"冒烟异常：{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
