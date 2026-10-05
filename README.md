# 木构文物两芯联合树轮对齐服务

将同一根梁材的两支树芯（A / B）**共同**对入地区年表，避免两支分别寻找
最佳位置后给出互相矛盾的砍伐年代。

- `POST /api/dendro/align`：两芯联合对齐
- `GET /health`：健康检查

## 问题模型

单支树芯在“样芯环 × 年表项”网格上寻找路径：

| 动作 | 含义 |
| --- | --- |
| 匹配 `(i,j)→(i+1,j+1)` | 样芯第 `i` 环对入年表第 `j` 项，要求 `|sample[i]-chron[j]| ≤ tolerance[i]` |
| 伪环 `(i,j)→(i+1,j)` | 跳过一个**样芯**环（年表中无对应年） |
| 缺环 `(i,j)→(i,j+1)` | 跳过一个**年表**项（该年样芯缺轮） |

约束：

- 两支**末环都必须匹配**，且落在**同一日历年**；
- 每支至少匹配 8 环；
- 首环与末环不允许被跳过，年表窗口前后的多余项不计缺环；
- 缺环、伪环各不超过该支给定限额（0–3）；
- 所有匹配差值不得越过逐环容差。

优化目标按字典序：

1. 两芯缺环与伪环**总数**最小；
2. 全部匹配差值的**最大绝对值**最小；
3. 全部匹配差值的**绝对值总和**最小。

对全部同优（三个目标都并列）映射判定 `unique` / `ambiguous`，并按
**映射序**（匹配对 `(样芯环, 年表项)` 序列的字典序）稳定返回前两份见证。

### 残失校准（可选 `terminal_loss`）

梁材表层常因刨削损失少量边材环，两支树芯的末个可见环可能不在同一年。
请求可附带 `terminal_loss`，为两芯分别给出 **0–3 的残失环数闭区间**
（启用时 `a`、`b` 两侧必须同时完整提供）：

- 每支末环仍须真实匹配年表；其匹配年份加选定残失数必须得到
  **同一砍伐年**；
- 残失环**不计入**缺环、伪环或原三级目标；
- 全部同优解释（含同一对映射的不同砍伐年取值）继续参与唯一性判定；
- 成功响应的 `common_end_year` 表示共同砍伐年，各芯映射保留实际末环
  年份 `end_year` 并返回 `outer_loss`（恒有
  `end_year + outer_loss == common_end_year`）；
- 前两份见证按（砍伐年, A 残失数, B 残失数, 原映射序）稳定排列；
- 若两芯可达末环与残失区间无法形成共同砍伐年，返回 `no_solution`，
  并在两芯证据中列出双方可达的推定砍伐年
  （`reachable_felling_years_within_limits`）。

未提供 `terminal_loss` 时，请求、响应、裁决与无解语义保持不变。

### 算法

- **阶段 A（求最优指标）**：帕累托动态规划。四元指标
  `(缺环, 伪环, 最大差值, 差值和)` 沿路径单调不减，严格支配者剪枝，
  指标向量相同只留一条代表（不影响最优值）。
- **阶段 B（枚举见证）**：固定最优终年与目标值后，以带可行性记忆化的
  DFS 按映射序惰性枚举，只取前两份；“唯一 / 歧义”判定精确。

算法实现在 `app/alignment.py`，并有 60 组随机算例与独立穷举参考实现
逐目标值、逐见证对照（`tests/test_alignment.py`）。

## 请求

```json
{
  "chronology": [/* 25–60 个整数生长指数 */],
  "start_year": 1990,
  "sample_a": [/* 10–24 环整数指数 */],
  "sample_b": [/* 10–24 环整数指数 */],
  "tolerance_a": [/* 与 sample_a 等长的非负逐环容差 */],
  "tolerance_b": [/* 与 sample_b 等长的非负逐环容差 */],
  "max_missing_a": 3,
  "max_false_a": 3,
  "max_missing_b": 3,
  "max_false_b": 3,
  "terminal_loss": {
    "a": {"min": 0, "max": 2},
    "b": {"min": 1, "max": 3}
  }
}
```

`terminal_loss` 可选；启用时 `a`、`b` 两侧必须同时完整提供，每侧为
0–3 的残失环数闭区间（`min ≤ max`）。字段不完整、区间倒置或越界均
返回定位明确的 422 校验错误；未提供该字段时请求与响应语义不变。

## 响应

成功（HTTP 200）：

```json
{
  "status": "unique | ambiguous",
  "common_end_year": 2009,
  "mappings": {
    "a": {
      "matched": [
        {"sample_ring": 1, "chronology_index": 9, "year": 1999,
         "sample_value": 63, "chronology_value": 63, "difference": 0}
      ],
      "missing_chronology_years": [2000],
      "missing_chronology_indices": [10],
      "false_sample_rings": [],
      "missing_count": 1, "false_count": 0, "match_count": 10,
      "end_year": 2009,
      "max_difference": 0, "difference_sum": 0
    },
    "b": { }
  },
  "objectives": {"total_skips": 1, "max_difference": 0, "difference_sum": 0},
  "witnesses": [ /* 按映射序稳定的前两份同优见证 */ ],
  "evidence": null
}
```

启用 `terminal_loss` 时：`common_end_year` 为共同砍伐年，各芯映射的
`end_year` 仍是实际末环年份，并各增 `outer_loss` 字段表示该芯残失环数；
见证按（砍伐年, 两芯残失数, 映射序）稳定排列。

无解（HTTP 200，业务状态 `no_solution`）：

```json
{
  "status": "no_solution",
  "common_end_year": null,
  "mappings": null,
  "objectives": null,
  "witnesses": [],
  "evidence": {
    "reason": "last_ring_unmatchable | insufficient_matchable_rings | skip_limits_exceeded | no_common_end_year | common_end_requires_more_skips",
    "shared_candidate_years": [],
    "cores": {
      "a": { "feasible_within_limits": false, "...": "两支各自的终年候选、限额内外最少跳环数等" },
      "b": { }
    },
    "shared_end_details": [ ]
  }
}
```

启用 `terminal_loss` 时，两芯证据各自增加
`reachable_felling_years_within_limits`（可达末环年 + 残失区间推出的
全部砍伐年），`limits` 中附 `outer_loss` 区间；`shared_candidate_years`
与 `shared_end_details` 中的年份均指推定砍伐年。

输入越界返回 HTTP 422：

```json
{ "status": "validation_error", "message": "请求参数未通过校验", "errors": [ ] }
```

## 运行

```bash
# 宿主机端口可配置（默认 8000）
APP_PORT=9090 docker compose up --build -d
curl http://localhost:9090/health
```

## verify 一次性校验服务

名为 `verify` 的一次性服务在应用**通过健康检查后**依次执行：

1. 代码测试（pytest）；
2. 应用构建检查（字节码编译、应用导入、健康探测）；
3. 两芯联合对齐 API 冒烟——兼容调用（唯一解 / 歧义 / 422 校验错误 /
   无解证据）与残失校准（共同砍伐年唯一解、无解时的双方可达推定年份、
   残失区间校验错误）。

并以自身退出码报告成败：

```bash
docker compose --profile verify up --build \
    --abort-on-container-exit --exit-code-from verify
# 成功退出码 0；任一环节失败非 0
```

也可在容器外直接运行（便于本地开发）：

```bash
pip install -r requirements.txt -r requirements-dev.txt
APP_BASE_URL=http://127.0.0.1:8000 sh scripts/verify.sh
```
