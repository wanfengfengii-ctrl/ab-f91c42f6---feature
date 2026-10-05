"""/api/dendro/align 与 /health 的接口测试。"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


BASE = dict(
    chronology=[100 + (i * 7) % 50 for i in range(30)],
    start_year=1990,
    sample_a=None,
    sample_b=None,
    tolerance_a=None,
    tolerance_b=None,
    max_missing_a=3, max_false_a=3,
    max_missing_b=3, max_false_b=3,
)


def _req():
    chron = BASE["chronology"]
    # A 芯精确落在下标 9..19（终年 19）；B 芯 10 环落在同区间、缺下标 11。
    sa = [chron[i] for i in range(9, 20)]
    sb = [chron[i] for i in [9, 10, 12, 13, 14, 15, 16, 17, 18, 19]]
    p = dict(BASE)
    p.update(
        sample_a=sa, sample_b=sb,
        tolerance_a=[0] * 11, tolerance_b=[0] * 10,
    )
    return p


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_align_ok():
    r = client.post("/api/dendro/align", json=_req())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "unique"
    assert body["common_end_year"] == 2009
    assert body["objectives"]["total_skips"] == 1
    assert body["mappings"]["b"]["missing_chronology_indices"] == [11]
    assert len(body["witnesses"]) == 1


def test_chronology_too_short():
    p = _req()
    p["chronology"] = [1] * 24
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "validation_error"
    assert any("chronology" in str(e.get("loc", "")) for e in body["errors"])


def test_chronology_too_long():
    p = _req()
    p["chronology"] = [1] * 61
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422


def test_sample_length_bounds():
    p = _req()
    p["sample_a"] = [1] * 9
    p["tolerance_a"] = [1] * 9
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422
    assert body_status(r) == "validation_error"


def test_tolerance_length_mismatch():
    p = _req()
    p["tolerance_a"] = [0] * 10      # sample_a 有 11 环
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422
    assert "tolerance_a" in r.text


def test_negative_tolerance_rejected():
    p = _req()
    p["tolerance_b"] = [-1] * 10
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422


def test_limit_out_of_range():
    p = _req()
    p["max_missing_a"] = 4
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422
    p = _req()
    p["max_false_b"] = -1
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422


def test_extra_field_rejected():
    p = _req()
    p["unexpected"] = 1
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422


def test_missing_field_rejected():
    p = _req()
    del p["sample_b"]
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422


def test_non_integer_rejected():
    p = _req()
    p["sample_a"] = [1.5] * 12
    p["tolerance_a"] = [0] * 12
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# terminal_loss（表层残失环）
# --------------------------------------------------------------------------- #

def _loss_req(la=0, ha=3, lb=0, hb=3):
    p = _req()
    p["terminal_loss"] = {"a": {"min": la, "max": ha},
                          "b": {"min": lb, "max": hb}}
    return p


def test_compat_response_has_no_outer_loss():
    # 未提供 terminal_loss 时响应形态保持不变。
    r = client.post("/api/dendro/align", json=_req())
    assert r.status_code == 200, r.text
    body = r.json()
    assert "outer_loss" not in body["mappings"]["a"]
    assert "terminal_loss" not in body
    for w in body["witnesses"]:
        assert "outer_loss" not in w["mappings"]["a"]
        assert "outer_loss" not in w["mappings"]["b"]


def test_terminal_loss_ok_felling_year():
    # 基础算例共同实际终年 2009；两侧各固定残失 1 -> 共同砍伐年 2010。
    r = client.post("/api/dendro/align", json=_loss_req(1, 1, 1, 1))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "unique"
    assert body["common_end_year"] == 2010
    ma, mb = body["mappings"]["a"], body["mappings"]["b"]
    assert ma["end_year"] == 2009 and ma["outer_loss"] == 1
    assert mb["end_year"] == 2009 and mb["outer_loss"] == 1
    # 残失不进入三级目标（仍是 B 芯 1 个缺环）
    assert body["objectives"]["total_skips"] == 1
    for w in body["witnesses"]:
        assert w["mappings"]["a"]["end_year"] + w["mappings"]["a"]["outer_loss"] \
            == w["common_end_year"]
        assert w["mappings"]["b"]["end_year"] + w["mappings"]["b"]["outer_loss"] \
            == w["common_end_year"]


def test_terminal_loss_zero_intervals_match_compat():
    r0 = client.post("/api/dendro/align", json=_req())
    r1 = client.post("/api/dendro/align", json=_loss_req(0, 0, 0, 0))
    assert r0.status_code == r1.status_code == 200
    b0, b1 = r0.json(), r1.json()
    assert b1["status"] == b0["status"]
    assert b1["common_end_year"] == b0["common_end_year"]
    assert b1["objectives"] == b0["objectives"]
    for side in ("a", "b"):
        m0 = dict(b0["mappings"][side])
        m1 = dict(b1["mappings"][side])
        assert m1.pop("outer_loss") == 0
        assert m1 == m0


def test_terminal_loss_witness_order():
    p = _loss_req(3, 3, 3, 3)
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] in ("unique", "ambiguous")
    keys = [
        (w["common_end_year"],
         w["mappings"]["a"]["outer_loss"],
         w["mappings"]["b"]["outer_loss"],
         tuple((m["sample_ring"], m["chronology_index"])
               for m in w["mappings"]["a"]["matched"]),
         tuple((m["sample_ring"], m["chronology_index"])
               for m in w["mappings"]["b"]["matched"]))
        for w in body["witnesses"]
    ]
    assert keys == sorted(keys)


def test_terminal_loss_no_solution_inferred_years():
    # 容差 0 下 A 实际终年 2014（下标 14）、B 实际终年 2025（下标 25），
    # 残失区间 0..3 推出的砍伐年集合 2014..2017 与 2025..2028 不相交。
    chron = list(range(100, 130))
    p = dict(BASE)
    p.update(
        chronology=chron, start_year=2000,
        sample_a=chron[3:15], sample_b=chron[16:26],
        tolerance_a=[0] * 12, tolerance_b=[0] * 10,
        terminal_loss={"a": {"min": 0, "max": 3},
                       "b": {"min": 0, "max": 3}},
    )
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "no_solution"
    ev = body["evidence"]
    assert ev["reason"] == "no_common_felling_year"
    assert ev["shared_candidate_years"] == []
    assert ev["cores"]["a"]["reachable_felling_years"] == [
        2014, 2015, 2016, 2017]
    assert ev["cores"]["b"]["reachable_felling_years"] == [
        2025, 2026, 2027, 2028]
    assert ev["cores"]["a"]["terminal_loss"] == {"min": 0, "max": 3}
    assert ev["cores"]["b"]["terminal_loss"] == {"min": 0, "max": 3}
    assert "shared_end_details" in ev


def test_terminal_loss_missing_side_rejected():
    # 只给 a 不给 b（或反之）：启用时两侧必须同时完整提供。
    p = _req()
    p["terminal_loss"] = {"a": {"min": 0, "max": 1}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "validation_error"
    assert any("terminal_loss" in ".".join(map(str, e.get("loc", [])))
               and "b" in ".".join(map(str, e.get("loc", [])))
               for e in body["errors"]), body["errors"]

    p = _req()
    p["terminal_loss"] = {"b": {"min": 0, "max": 1}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422
    assert any("terminal_loss" in ".".join(map(str, e.get("loc", [])))
               and "a" in ".".join(map(str, e.get("loc", [])))
               for e in r.json()["errors"])


def test_terminal_loss_missing_bound_rejected():
    # 单侧缺少 min / max：定位到具体字段。
    p = _req()
    p["terminal_loss"] = {"a": {"max": 1}, "b": {"min": 0, "max": 1}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422
    locs = [".".join(map(str, e.get("loc", []))) for e in r.json()["errors"]]
    assert any("terminal_loss" in s and "a" in s and "min" in s for s in locs), locs

    p = _req()
    p["terminal_loss"] = {"a": {"min": 0, "max": 1},
                          "b": {"min": 0, "max": 3}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 200


def test_terminal_loss_inverted_interval_rejected():
    p = _loss_req(2, 1, 0, 1)        # A: min > max
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422
    body = r.json()
    locs = [".".join(map(str, e.get("loc", []))) for e in body["errors"]]
    assert any("terminal_loss" in s and "a" in s for s in locs), locs
    assert any("倒置" in e.get("msg", "") for e in body["errors"])

    p = _loss_req(0, 3, 3, 0)        # B: min > max
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422
    assert any("倒置" in e.get("msg", "") for e in r.json()["errors"])


def test_terminal_loss_bound_out_of_range():
    for val in (-1, 4):
        p = _loss_req(val, val, 0, 1)
        r = client.post("/api/dendro/align", json=p)
        assert r.status_code == 422
    p = _req()
    p["terminal_loss"] = {"a": {"min": 0, "max": 1.5},
                          "b": {"min": 0, "max": 1}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422


def test_terminal_loss_extra_field_rejected():
    p = _req()
    p["terminal_loss"] = {"a": {"min": 0, "max": 1, "extra": 1},
                          "b": {"min": 0, "max": 1}}
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 422


def test_no_solution_shape():
    p = _req()
    p["sample_b"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 99999]
    p["tolerance_b"] = [0] * 10
    r = client.post("/api/dendro/align", json=p)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no_solution"
    assert body["common_end_year"] is None
    assert body["mappings"] is None
    ev = body["evidence"]
    assert ev["reason"] == "last_ring_unmatchable"
    assert "a" in ev["cores"] and "b" in ev["cores"]
    assert ev["cores"]["b"]["last_ring_candidate_years"] == []
    assert "shared_end_details" in ev


def body_status(r):
    return r.json()["status"]
