#!/usr/bin/env sh
# 一次性校验服务：在应用健康后依次执行
#   1) 代码测试（pytest）
#   2) 应用构建检查（字节码编译 + 应用导入 + 健康探测）
#   3) 两芯联合对齐 API 冒烟
# 任一步失败即以非零退出码上报。
set -eu

BASE_URL="${APP_BASE_URL:-http://app:8000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> [1/3] 代码测试"
python -m pytest -q tests

echo "==> [2/3] 应用构建检查"
python -m compileall -q app
python -c "from app.main import app; print('app import ok:', app.title)"

echo "    等待应用健康: ${BASE_URL}/health"
i=0
while [ "$i" -lt 30 ]; do
    if python - "$BASE_URL" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1] + "/health", timeout=2) as r:
        ok = r.status == 200 and json.load(r)["status"] == "ok"
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
    then
        echo "    应用已健康"
        break
    fi
    i=$((i + 1))
    sleep 1
done
if [ "$i" -ge 30 ]; then
    echo "!! 应用未在限定时间内变健康" >&2
    exit 1
fi

echo "==> [3/3] 两芯联合对齐 API 冒烟"
python "$ROOT/scripts/smoke.py" "$BASE_URL"

echo ""
echo "全部校验通过：测试 / 构建 / 联合对齐冒烟。"
