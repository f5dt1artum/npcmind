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

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

当前基线提供行为树的一次性求值与有限状态机的单步推进；寻路等能力仍留给后续任务从已冻结事实出发独立设计并验证。
