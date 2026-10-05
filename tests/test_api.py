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
