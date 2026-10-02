# NpcMind

这是一个面向游戏 AI的游戏 AI 与 NPC 智能行为引擎。长期目标是提供行为树与黑板、层次状态机、目标导向行动规划、效用决策、寻路与避障、感知与记忆、对话意图和难度自适应，把 NPC 行为沉淀为可复用引擎。

仓库采用 Python，当前冻结基线只提供进程健康检查。后续能力必须通过独立题目逐步实现；每个题目都应定义可观察的公共行为、兼容边界和失败语义，不得依赖未公开内部 API。

## 启动

```bash
PYTHONPATH=src python3 -m npcmind.server --host 127.0.0.1 --port 8080
```

服务默认监听 `127.0.0.1:8080`，可通过 `NPCMIND_ADDR` 修改。`GET /healthz` 返回 JSON 健康状态。

`POST /v1/behavior-trees/evaluate` 执行一次行为树 tick：请求体为 `{"tree": ..., "blackboard": {...}}`（黑板可省略，视为空对象），返回根节点 status、更新后的 blackboard 和按访问顺序记录的 trace。节点类型支持 `sequence`、`selector`、`condition`（`exists` / `equals` / `not_equals`）与 `action`（`set` / `delete` / `status`）。JSON 无法解析返回 400 `invalid_json`；树结构或字段非法返回 422 `invalid_tree`。Python 侧等价入口为 `npcmind.service.Service.evaluate_behavior(request)`，同类错误抛出 `ValueError`。

`POST /v1/state-machines/step` 推进有限状态机一步：请求体为 `{"machine": ..., "event": "...", "current_state": "...", "blackboard": {...}}`，其中 `current_state` 省略时使用机器的 `initial`，`blackboard` 省略时视为空对象，调用不会修改传入对象。machine 需定义非空 `states`、`initial` 和有序 `transitions`；状态与迁移使用非空且唯一的字符串 `id`，迁移的 `from`、`to` 必须引用已声明状态。每次调用只处理一个字符串事件，按声明顺序考察 `from` 等于当前状态且 `event` 精确匹配的迁移；迁移可带一个 `condition`（沿用行为树 `exists` / `equals` / `not_equals` 与 JSON 类型敏感比较语义，省略时视为满足），选中首个条件成功的迁移后按顺序执行仅含 `set` / `delete` 的 `actions`。返回 `previous_state`、`state`、`transition`（命中的迁移 id，未命中为 `null`）、`blackboard` 和 `trace`（按考察顺序记录候选 id 与条件结果）；未命中不是错误，状态与黑板保持不变。机器和请求在执行任何动作前完整校验；JSON 无法解析返回 400 `invalid_json`，结构或字段非法返回 422 `invalid_state_machine`。Python 侧等价入口为 `npcmind.service.Service.step_state_machine(request)`，同类错误抛出 `ValueError`。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

当前基线提供行为树的一次性求值与有限状态机的单步推进；寻路等能力仍留给后续任务从已冻结事实出发独立设计并验证。
