# NpcMind

这是一个面向游戏 AI的游戏 AI 与 NPC 智能行为引擎。长期目标是提供行为树与黑板、层次状态机、目标导向行动规划、效用决策、寻路与避障、感知与记忆、对话意图和难度自适应，把 NPC 行为沉淀为可复用引擎。

仓库采用 Python，当前冻结基线只提供进程健康检查。后续能力必须通过独立题目逐步实现；每个题目都应定义可观察的公共行为、兼容边界和失败语义，不得依赖未公开内部 API。

## 启动

```bash
PYTHONPATH=src python3 -m npcmind.server --host 127.0.0.1 --port 8080
```

服务默认监听 `127.0.0.1:8080`，可通过 `NPCMIND_ADDR` 修改。`GET /healthz` 返回 JSON 健康状态。

`POST /v1/behavior-trees/evaluate` 执行一次行为树 tick：请求体为 `{"tree": ..., "blackboard": {...}}`（黑板可省略，视为空对象），返回根节点 status、更新后的 blackboard 和按访问顺序记录的 trace。节点类型支持 `sequence`、`selector`、`condition`（`exists` / `equals` / `not_equals`）与 `action`（`set` / `delete` / `status`）。JSON 无法解析返回 400 `invalid_json`；树结构或字段非法返回 422 `invalid_tree`。Python 侧等价入口为 `npcmind.service.Service.evaluate_behavior(request)`，同类错误抛出 `ValueError`。

`POST /v1/state-machines/step` 将有限状态机推进一个事件：请求体为 `{"machine": ..., "event": "...", "current_state": "...", "blackboard": {...}}`（`current_state` 省略时取 `initial`，`blackboard` 省略视为空对象，传入对象不会被修改）。`machine` 需定义非空 `states`、`initial` 与有序 `transitions`；状态与迁移使用非空且唯一的字符串 id，迁移的 `from` / `to` 必须引用已声明状态。每次调用只处理一个字符串事件，按声明顺序考察 `from` 等于当前状态且 `event` 精确匹配的迁移，选中首个 `condition`（与行为树相同的 `exists` / `equals` / `not_equals` 语义，省略视为满足）成立的迁移，并顺序执行仅含 `set` / `delete` 的 `actions`。返回 `previous_state`、`state`、`transition`（未命中为 `null`）、`blackboard` 与按考察顺序记录候选 id 及条件结果的 `trace`；未命中时状态与黑板不变。机器与请求在执行动作前完整校验，非法时返回 422 `invalid_state_machine`，Python 侧等价入口 `Service.step_state_machine(request)` 抛出 `ValueError`。

`POST /v1/goap/plan` 执行一次性目标导向行动规划：请求体为 `{"world": {...}, "goal": {...}, "actions": [...]}`，`world` 与 `goal` 为以非空字符串作键的 JSON 对象；每个行动含全局唯一的非空字符串 `id`、可省略的 `cost`（默认 1，仅允许正整数）、可省略的 `preconditions` 与 `effects`（省略视为空对象）。前置条件仅在当前状态存在同名键且值相等时成立；效果覆盖对应键，其余状态项保留；目标的全部键存在且相等即满足（世界可含额外键）。值比较沿用现有规则：布尔值不等于数字，整数与浮点数按数值比较，对象与数组递归比较。初始状态已满足目标时返回 `status` 为 `SUCCESS`、空 `plan`、`cost` 为 0 及原始 `final_world`；否则返回总成本最低的计划（`plan` 按执行顺序列出行动 id，`cost` 为总成本，`final_world` 为执行后状态），成本相同时按行动在输入列表中的位置序列作字典序比较并取最小者。目标不可达时返回 `status` 为 `UNREACHABLE`、空 `plan`、`cost` 为 `null` 及原始 `world` 作为 `final_world`。请求在搜索前完整校验，非法时返回 422 `invalid_goap`，Python 侧等价入口 `Service.plan_goap(request)` 抛出 `ValueError`；调用不保存跨请求状态，也不修改传入的请求及其嵌套对象。

`POST /v1/navigation/path` 执行一次性二维方格寻路：请求体为 `{"grid": [[...]], "start": {"x": 0, "y": 0}, "goal": {"x": 2, "y": 1}}`。`grid` 是非空矩形二维数组，每格只能是 `null`（不可通行）或正整数（进入该格的代价，布尔值不视为整数）；坐标以左上角为原点，`x` 向右、`y` 向下，移动仅允许上下左右相邻的可通行格。成功时返回 `status` 为 `SUCCESS`、包含两端点且按移动顺序排列的 `path`（每个元素为同样的 `x`/`y` 坐标对象），以及不计起点、按进入后续格代价累加的 `cost`。结果选择总成本最低的路线；多条路线成本相同时，将每条路线转换为按行列坐标组成的完整序列，并按各坐标的 `y`、`x` 顺序作字典序比较，返回最小者。起点等于终点时返回只含该点的 `path` 和 0 成本；目标不可达不是请求错误，返回 `status` 为 `UNREACHABLE`、空 `path` 与 `null` 成本。请求在搜索前完整校验：请求或坐标不是对象、`grid` 为空或不规则、格值不合法、坐标字段缺失或含布尔值、坐标越界、起点或终点落在不可通行格上，均返回 422 `invalid_navigation`，Python 侧等价入口 `Service.find_path(request)` 抛出 `ValueError`，且不返回部分路线；调用不保存跨请求地图或搜索状态，也不修改传入的请求及其嵌套值。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

当前基线提供行为树的一次性求值、有限状态机的单步推进、一次性 GOAP 规划与无状态二维方格寻路；效用决策、避障、感知记忆等能力仍留给后续任务从已冻结事实出发独立设计并验证。
