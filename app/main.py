"""两芯联合树轮对齐 Web 服务。"""

from __future__ import annotations

from typing import List

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from .alignment import align_joint


def _sanitize_errors(errors: list[dict]) -> list[dict]:
    """Pydantic v2 的 ctx 可能携带不可 JSON 序列化的异常对象，统一清洗。"""
    clean = []
    for e in errors:
        item = {k: v for k, v in e.items() if k != "ctx"}
        ctx = e.get("ctx")
        if ctx:
            item["ctx"] = {
                k: str(v) if not isinstance(
                    v, (str, int, float, bool, type(None), list, dict)
                ) else v
                for k, v in ctx.items()
            }
        clean.append(item)
    return clean


class AlignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chronology: List[int] = Field(
        ..., description="地区年表生长指数（整数），长度 25–60。")
    start_year: int = Field(
        ..., description="年表首项对应的日历年。")
    sample_a: List[int] = Field(..., min_length=10, max_length=24,
                                description="A 芯生长指数序列，长度 10–24。")
    sample_b: List[int] = Field(..., min_length=10, max_length=24,
                                description="B 芯生长指数序列，长度 10–24。")
    tolerance_a: List[int] = Field(
        ..., description="A 芯逐环容差（非负整数），与 sample_a 等长。")
    tolerance_b: List[int] = Field(
        ..., description="B 芯逐环容差（非负整数），与 sample_b 等长。")
    max_missing_a: int = Field(..., ge=0, le=3, description="A 芯缺环限额 0–3。")
    max_false_a: int = Field(..., ge=0, le=3, description="A 芯伪环限额 0–3。")
    max_missing_b: int = Field(..., ge=0, le=3, description="B 芯缺环限额 0–3。")
    max_false_b: int = Field(..., ge=0, le=3, description="B 芯伪环限额 0–3。")

    @field_validator("chronology")
    @classmethod
    def _check_chronology(cls, v: List[int]) -> List[int]:
        if not 25 <= len(v) <= 60:
            raise ValueError("chronology 长度必须在 25 至 60 之间")
        return v

    @field_validator("sample_a", "sample_b")
    @classmethod
    def _check_sample(cls, v: List[int]) -> List[int]:
        if not 10 <= len(v) <= 24:
            raise ValueError("样芯序列长度必须在 10 至 24 之间")
        return v

    @field_validator("tolerance_a", "tolerance_b")
    @classmethod
    def _check_tol_nonneg(cls, v: List[int]) -> List[int]:
        if any(x < 0 for x in v):
            raise ValueError("逐环容差必须为非负整数")
        return v

    @model_validator(mode="after")
    def _check_tol_lengths(self) -> "AlignRequest":
        if len(self.tolerance_a) != len(self.sample_a):
            raise ValueError("tolerance_a 长度必须与 sample_a 相同")
        if len(self.tolerance_b) != len(self.sample_b):
            raise ValueError("tolerance_b 长度必须与 sample_b 相同")
        return self


app = FastAPI(
    title="木构文物两芯联合树轮对齐服务",
    version="1.0.0",
    description=(
        "将同一根梁材的两支树芯共同对入地区年表：先最小化两芯缺环与伪环"
        "总数，再依次最小化最大差值与差值总和，并对同优映射判定唯一 / 歧义。"
    ),
)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "status": "validation_error",
            "message": "请求参数未通过校验",
            "errors": _sanitize_errors(exc.errors()),
        },
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/dendro/align")
def align(req: AlignRequest) -> dict:
    return align_joint(req.model_dump())
