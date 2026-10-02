# NpcMind

这是一个面向游戏 AI的游戏 AI 与 NPC 智能行为引擎。长期目标是提供行为树与黑板、层次状态机、目标导向行动规划、效用决策、寻路与避障、感知与记忆、对话意图和难度自适应，把 NPC 行为沉淀为可复用引擎。

仓库采用 Python，当前冻结基线只提供进程健康检查。后续能力必须通过独立题目逐步实现；每个题目都应定义可观察的公共行为、兼容边界和失败语义，不得依赖未公开内部 API。

## 启动

```bash
PYTHONPATH=src python3 -m npcmind.server --host 127.0.0.1 --port 8080
```

服务默认监听 `127.0.0.1:8080`，可通过 `NPCMIND_ADDR` 修改。`GET /healthz` 返回 JSON 健康状态。

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

当前基线刻意不包含行为树、状态机与寻路的实现，以便后续任务从已冻结事实出发独立设计并验证这些能力。
