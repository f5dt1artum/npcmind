# NpcMind

这是一个面向游戏 AI的游戏 AI 与 NPC 智能行为引擎。长期目标是提供行为树与黑板、层次状态机、目标导向行动规划、效用决策、寻路与避障、感知与记忆、对话意图和难度自适应，把 NPC 行为沉淀为可复用引擎。

仓库采用 Python，当前冻结基线只提供进程健康检查。后续能力必须通过独立题目逐步实现；每个题目都应定义可观察的公共行为、兼容边界和失败语义，不得依赖未公开内部 API。

## 启动

```bash
PYTHONPATH=src python3 -m npcmind.server --host 127.0.0.1 --port 8080
```

服务默认监听 `127.0.0.1:8080`，可通过 `NPCMIND_ADDR` 修改。`GET /healthz` 返回 JSON 健康状态。

`POST /v1/behavior-trees/evaluate` 执行一次行为树 tick：请求体为 `{"tree": ..., "blackboard": {...}}`（黑板可省略，视为空对象），返回根节点 status、更新后的 blackboard 和按访问顺序记录的 trace。节点类型支持 `sequence`、`selector`、`condition`（`exists` / `equals` / `not_equals`）与 `action`（`set` / `delete` / `status`）。JSON 无法解析返回 400 `invalid_json`；树结构或字段非法返回 422 `invalid_tree`。Python 侧等价入口为 `npcmind.service.Service.evaluate_behavior(request)`，同类错误抛出 `ValueError`。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

当前基线提供行为树与黑板的一次性求值；状态机、寻路等能力仍留给后续任务从已冻结事实出发独立设计并验证。
