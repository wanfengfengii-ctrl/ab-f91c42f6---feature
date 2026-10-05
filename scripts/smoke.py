#!/usr/bin/env python3
"""两芯联合对齐 API 冒烟脚本。

覆盖四类响应：唯一解、歧义解、越界校验错误、无解（含两芯证据）。
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
