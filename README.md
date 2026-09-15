# Enterprise-Agent——企业可治理多智能体 Agent 应用平台

> 面向企业复杂任务执行场景，基于 **LangChain + LangGraph** 构建可治理、可恢复的 Multi-Agent Runtime，围绕 **Agent Runtime、Tool Governance、Stateful Workflow、Memory 与 MCP Integration** 统一编排 Agent、Tool 与执行状态。

---

## 1. Overview

Enterprise-Agent 不是面向单轮问答的普通 Agent，而是面向企业任务执行场景构建的 **Agent Runtime**。

系统将 Agent 执行拆分为：

```text
User Request
     │
     ▼
Runtime Context
     │
     ▼
Memory Retrieval
     │
     ▼
Supervisor
     │
     ├──────────────┬──────────────┬──────────────┐
     ▼              ▼              ▼              ▼
Knowledge       Operations       Ticket       Research
Agent            Agent           Agent          Agent
     │              │              │              │
     └──────────────┴──────────────┴──────────────┘
                    │
                    ▼
          Specialist Outcome Router
              │              │
           completed       failed
              │              │
              ▼              ▼
          Supervisor      Recovery
                              │
                       Retry / Re-route
                              │
                              ▼
                         Supervisor
                              │
                              ▼
                    Workflow Completion
                              │
                     ┌────────┴────────┐
                     ▼                 ▼
               Memory Persist       Finalizer
                                         │
                                         ▼
                                        END
```

核心目标不是让 Agent “更加自主”，而是让 Agent 的自主执行处于明确的 **State、Policy、Tool 与 Workflow 控制边界**之内。

---

## 2. Design Goals

项目围绕以下几个目标设计：

| 目标       | 设计                                              |
| -------- | ----------------------------------------------- |
| Agent 协作 | Supervisor–Specialist Stateful Workflow         |
| 工具治理     | Tool Registry + Permission Policy + Risk Policy |
| 动态工具暴露   | 根据 Runtime Context 控制 Specialist 可见 Tool        |
| 高风险操作控制  | Human-in-the-Loop                               |
| 执行可靠性    | Tool Retry + Workflow Recovery                  |
| 状态连续性    | LangGraph State + PostgreSQL Checkpoint         |
| 长期上下文    | Long-term Memory                                |
| 外部能力接入   | MCP Client / Server / Adapter                   |
| 企业知识访问   | Enterprise-RAG Tool Integration                 |
| 数据安全     | PII Middleware + Runtime Context                |

---

# 3. Architecture

## 3.1 Runtime Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                     Enterprise Agent Runtime               │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Runtime Context                                           │
│  ├── user_id                                               │
│  ├── user_role                                             │
│  └── tenant_id                                             │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 Stateful Workflow                   │  │
│  │                                                      │  │
│  │  Memory Retrieval                                    │  │
│  │        ↓                                             │  │
│  │  Supervisor                                           │  │
│  │        ↓                                             │  │
│  │  Specialist Agents                                    │  │
│  │        ↓                                             │  │
│  │  Outcome / Conditional Routing                       │  │
│  │        ↓                                             │  │
│  │  Recovery                                             │  │
│  │        ↓                                             │  │
│  │  Supervisor / Finalizer                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   Tool Governance                   │  │
│  │                                                      │  │
│  │ ToolRegistry → PermissionPolicy → RiskPolicy         │  │
│  │       ↓                 ↓              ↓             │  │
│  │ Dynamic Exposure     Access Check     HITL           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  Middleware Pipeline                 │  │
│  │                                                      │  │
│  │ Logging → PII → HITL → Tool Error → Retry           │  │
│  │                                                → Memory │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────────────┐    ┌────────────────────────┐   │
│  │ Long-term Memory     │    │ MCP Integration        │   │
│  │ Model / Store /      │    │ Client → Adapter →     │   │
│  │ Embedder / Extractor │    │ Tool Registry           │   │
│  │ / Manager            │    │                          │   │
│  └──────────────────────┘    └────────────────────────┘   │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

# 4. Stateful Multi-Agent Workflow

系统基于 LangGraph 构建状态化 Workflow。

核心状态包括：

```text
messages
retrieved_memories
current_agent
task_status
workflow_complete
completed_agents
handoff_reason
handoff_task
tool_results
approval_required
approval_status
approval_events
error
last_failed_node
last_failed_tool
recovery_attempts
recovery_status
recovery_reason
resume_required
final_answer
execution_summary
```

其中：

* **Runtime Context** 保存当前请求的可信身份与权限信息；
* **Graph State** 保存 Workflow 执行过程中不断变化的数据；
* **Long-term Memory** 保存跨 Workflow 的可持久化业务记忆；
* **Checkpoint** 持久化 Graph State，使长流程能够从中断点恢复。

`user_id / user_role / tenant_id` 不直接写入 Graph State，而由 `EnterpriseAgentContext` 提供，用于权限控制、Memory Scope、Logging 与数据隔离。

---

## 4.1 Supervisor

Supervisor 是 Workflow 的控制节点，主要负责：

```text
User Request
      ↓
Task Understanding
      ↓
Specialist Routing
      ↓
Result Assessment
      ↓
Re-planning / Completion Decision
```

Supervisor 不直接执行业务 Tool。

它只负责：

* 分析当前任务；
* 选择合适的 Specialist；
* 生成 handoff task；
* 根据 Specialist 返回结果决定下一步；
* 判断 Workflow 是否真正完成；
* 在异常情况下配合 Recovery 重新规划。

一个重要的设计约束是：

> **Specialist completion ≠ Workflow completion**

Specialist 完成自己的 delegated task 后必须回到 Supervisor，由 Supervisor 决定整个 Workflow 是否已经完成。

---

# 5. Specialist Agents

当前系统包含四类 Specialist：

| Specialist         | 职责                 | Tool                                             |
| ------------------ | ------------------ | ------------------------------------------------ |
| `knowledge_agent`  | 企业内部知识检索           | `rag_search`                                     |
| `operations_agent` | 企业服务健康检查           | `get_service_health`                             |
| `ticket_agent`     | 企业工单 / Incident 操作 | `get_ticket` / `create_ticket` / `update_ticket` |
| `research_agent`   | 外部公开信息研究           | `web_search`                                     |

Specialist 采用 **restricted tool scope**，只暴露完成当前领域任务所需的 Tool。

例如：

```text
knowledge_agent
    ↓
rag_search

operations_agent
    ↓
get_service_health

ticket_agent
    ↓
get_ticket
create_ticket
update_ticket

research_agent
    ↓
web_search
```

这样可以减少无关 Tool 暴露，同时降低错误调用和不必要的上下文负担。

---

# 6. Conditional Routing

Workflow 支持基于实际执行结果的 Conditional Routing，而不是单纯按照固定顺序执行 Agent。

典型场景：

```text
User
 │
 ▼
operations_agent
 │
 │ get_service_health
 ▼
status = degraded
 │
 ▼
Supervisor
 │
 ▼
ticket_agent
 │
 ▼
create_ticket
```

例如：

```text
Please check the current health of payment-service.
If the service is degraded, create a P1 Incident ticket.
```

执行过程中：

```text
payment-service
      ↓
get_service_health
      ↓
degraded
      ↓
ticket_agent
      ↓
create_ticket
```

而不是预先固定：

```text
operations_agent
→ ticket_agent
```

因此后续 Agent 是否执行由**实际业务状态 + Workflow State**决定。

---

# 7. Tool Governance

Agent 不直接拥有无约束的 Tool 集合，而是经过统一的 Tool Governance Layer。

```text
Tool
 │
 ▼
ToolMetadata
 │
 ▼
ToolRegistry
 │
 ├── PermissionPolicy
 │
 ├── Dynamic Tool Exposure
 │
 └── RiskPolicy
        │
        ▼
 Human-in-the-Loop
        │
        ▼
 Tool Execution
```

---

## 7.1 ToolMetadata

统一描述 Tool：

```text
name
category
permissions
tags
source
risk_level
description
```

Tool 来源统一抽象为：

```text
LOCAL
MCP
```

从而让本地 Tool 与外部 MCP Tool 能够进入统一治理链路。

---

## 7.2 ToolRegistry

`ToolRegistry` 是 Runtime 的 Tool Catalog。

负责：

```text
Tool Registration
Tool Lookup
Tool Metadata
Tool Scope
```

所有 Tool 在运行时进入 Registry 后，统一参与 Permission、Risk 与 Exposure 控制。

---

## 7.3 PermissionPolicy

PermissionPolicy 基于 Runtime Context 判断：

```text
user_role
    +
tool
    ↓
Permission Decision
```

当前角色包括：

```text
employee
developer
admin
```

例如：

```text
employee
    ↓
restricted tools

developer
    ↓
development tools

admin
    ↓
full permitted tools
```

Permission 是 Tool 执行前的访问边界，而不是 Agent Prompt 中的软约束。

---

# 8. Risk Policy + Human-in-the-Loop

对于具有外部副作用的 Tool，系统进一步执行 Risk Evaluation。

```text
Tool Call
   │
   ▼
RiskPolicy
   │
   ├── Low / No Approval
   │
   └── High Risk
          │
          ▼
        HITL
```

当前高风险场景以 Tool Call Approval 为核心。

HITL 支持：

```text
Approve
Reject
Edit
```

---

## 8.1 Approve

```text
Tool Call
   ↓
HITL
   ↓
Approve
   ↓
Execute
```

---

## 8.2 Reject

```text
Tool Call
   ↓
HITL
   ↓
Reject
   ↓
Tool 不执行
   ↓
Workflow 继续处理
```

---

## 8.3 Edit

```text
Tool Call
   ↓
HITL
   ↓
Edit
   ↓
Human 修改 Tool Arguments
   ↓
重新提交
   ↓
Execute
```

例如：

```text
Original:
{
  "title": "[P1] payment-service degraded",
  "priority": "P1"
}
```

Human 修改：

```json
{
  "title": "edited integration test",
  "priority": "P1"
}
```

最终 Tool 使用修改后的参数执行。

---

# 9. Middleware Runtime

Runtime 使用统一 Middleware Pipeline 管理 Agent 生命周期。

当前主要包含：

```text
Logging
PII
Human-in-the-Loop
Tool Error
Tool Retry
Model Retry
Memory
```

整体执行模型：

```text
Agent
  ↓
Middleware Pipeline
  ↓
Tool Calling / Model Calling
  ↓
Execution Result
```

Middleware 将横切逻辑从具体 Agent 中抽离，使不同 Specialist 可以共享相同的执行控制机制。

---

# 10. Retry & Recovery

项目将**局部 Tool Retry**与**Workflow-level Recovery**明确分层。

## 10.1 Tool Retry

解决单个 Tool 调用瞬时失败：

```text
Tool Call
   ↓
Retry
   ↓
Retry
   ↓
Retry
   ↓
Success / Error
```

Tool Retry 属于执行层机制。

---

## 10.2 Tool Error

当 Tool 超过 Retry 次数仍然失败：

```text
Tool
 ↓
Retry
 ↓
Failure
 ↓
Tool Error
```

错误信息进入 Workflow State。

---

## 10.3 Workflow Recovery

Workflow-level Recovery 不再简单重复同一个 Tool，而是重新处理 Workflow 状态：

```text
Specialist Failure
       ↓
Recovery
       │
       ├── retry
       │
       ├── reroute
       │
       ├── human_review
       │
       └── terminal failure
```

其中：

### Retry

重新执行当前 Specialist。

### Re-route

把控制权重新交给 Supervisor，由 Supervisor 根据新的 State 重新规划。

### Human Review

Workflow 级别暂停，由人工决定：

```text
retry
reroute
reject
```

### Terminal Failure

超过最大 Recovery 次数后：

```text
workflow_complete = False
task_status = failed
```

并保存：

```text
last_failed_node
last_failed_tool
recovery_attempts
recovery_reason
```

---

# 11. Checkpoint / Resume

Workflow 使用 LangGraph Checkpointer 保存执行状态。

核心模型：

```text
Workflow
   ↓
Checkpoint
   ↓
Thread
```

因此在：

```text
Tool HITL
Workflow HITL
Recovery
```

等场景发生 interrupt 后，可以：

```text
interrupt
   ↓
Human Decision
   ↓
Command(resume=...)
   ↓
Restore State
   ↓
Continue Workflow
```

这使 Workflow 具备长流程状态连续性，而不是一次性函数调用。

---

# 12. Long-term Memory

长期记忆独立于 Graph State。

当前架构：

```text
                Long-term Memory
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      Model          Store         Embedder
        │              │
        └──────┬───────┘
               ▼
          Extractor
               │
               ▼
           Manager
               │
               ▼
       Memory Middleware
```

核心组件：

```text
MemoryModel
MemoryStore
MemoryEmbedder
MemoryExtractor
MemoryManager
```

Workflow 开始：

```text
Runtime Context
      ↓
user_id
      ↓
Memory Retrieval
      ↓
retrieved_memories
      ↓
Graph State
```

后续 Specialist 在整个 Workflow 中共享 `retrieved_memories`，避免每一次模型调用都重复查询长期记忆。

Workflow 结束后，根据记忆策略决定是否执行：

```text
Memory Persistence
```

---

# 13. MCP Integration

系统将 MCP Tool 纳入统一 Tool Runtime。

整体链路：

```text
MCP Server
    │
    ▼
MCP Client
    │
    ├── initialize
    ├── session
    └── list_tools
    │
    ▼
MCP Adapter
    │
    ▼
LangChain Structured Tool
    │
    ▼
ToolRegistry
    │
    ▼
Permission / Risk / HITL
    │
    ▼
Agent Execution
```

MCP Tool 不直接暴露给 Agent，而是首先进入 Runtime Governance Layer。

因此本地 Tool 和 MCP Tool 最终共享：

```text
Metadata
Registry
Permission
Risk
Dynamic Exposure
Middleware
HITL
Retry
```

---

# 14. Enterprise-RAG Integration

`knowledge_agent` 不直接实现 RAG Pipeline，而是通过统一 Tool 接口调用独立的 Enterprise-RAG 服务。

```text
Knowledge Agent
      ↓
rag_search
      ↓
HTTP
      ↓
Enterprise-RAG
      ↓
Retrieval + Correction
      ↓
Evidence Documents
      ↓
Knowledge Agent
```

Enterprise-RAG 负责：

```text
Query Understanding
Hybrid Retrieval
Parent-Child Retrieval
Reranking
Coverage Evaluation
Corrective Retrieval
```

Enterprise-Agent 只负责：

```text
Agent Orchestration
Tool Governance
Workflow Execution
```

因此两个项目保持清晰边界：

```text
Enterprise-RAG
    = Retrieval Engineering

Enterprise-Agent
    = Agent Runtime / Orchestration
```

---

# 15. End-to-End Example

下面展示一个完整的企业任务执行过程：

```text
User
 │
 │ Please check payment-service.
 │ If degraded, create a P1 Incident.
 ▼
Memory Retrieval
 │
 ▼
Supervisor
 │
 ▼
operations_agent
 │
 ▼
get_service_health
 │
 └── status = degraded
        │
        ▼
     Supervisor
        │
        ▼
     ticket_agent
        │
        ▼
     create_ticket
        │
        ▼
   RiskPolicy / HITL
        │
        ▼
      Approve
        │
        ▼
    Tool Retry Layer
        │
        ▼
    create_ticket
        │
        ▼
      Success
        │
        ▼
     Supervisor
        │
        ▼
 Workflow Complete
        │
        ├──────────────┐
        ▼              ▼
 Memory Persist     Finalizer
                         │
                         ▼
                        END
```

---

# 16. Failure Recovery Example

当 `create_ticket` 连续失败时：

```text
create_ticket
   ↓
attempt 1 ✗
   ↓
attempt 2 ✗
   ↓
attempt 3 ✗
   ↓
Tool Error
   ↓
Recovery
   ↓
reroute / retry
   ↓
ticket_agent
   ↓
create_ticket
   ↓
HITL
   ↓
retry
   ↓
...
   ↓
Maximum Recovery Attempts
   ↓
Workflow Failed
```

最终 State 保留：

```text
task_status = failed
workflow_complete = False
recovery_attempts = 2
recovery_status = failed
last_failed_node = ticket_agent
last_failed_tool = create_ticket
resume_required = False
```

这一区分：

```text
Tool Retry
        ≠
Workflow Recovery
```

是 Runtime 可靠性设计的重要部分。

---

# 17. Runtime Context vs Graph State vs Memory

三者严格区分：

| 数据                   | 所属              | 作用                     |
| -------------------- | --------------- | ---------------------- |
| `user_id`            | Runtime Context | 用户身份                   |
| `user_role`          | Runtime Context | 权限控制                   |
| `tenant_id`          | Runtime Context | 租户边界                   |
| `messages`           | Graph State     | 当前会话                   |
| `tool_results`       | Graph State     | 当前 Workflow 执行结果       |
| `recovery_*`         | Graph State     | 当前 Workflow 恢复状态       |
| `retrieved_memories` | Graph State     | 本次 Workflow 使用的 Memory |
| Long-term Memory     | Memory Store    | 跨 Workflow 持久化信息       |

核心原则：

> **Identity / Authorization 属于 Runtime Context，Execution State 属于 Graph State，跨会话信息属于 Long-term Memory。**

---

---

# 18. Main Runtime Flow

```text
                 ┌──────────────────┐
                 │   User Request   │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │ Runtime Context  │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │ Memory Retrieval │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │    Supervisor    │
                 └────────┬─────────┘
                          ▼
              ┌────────────────────────┐
              │  Specialist Selection  │
              └────────────┬───────────┘
                           ▼
                ┌─────────────────────┐
                │  Specialist Agent   │
                └──────────┬──────────┘
                           ▼
                   Tool Governance
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         Permission      Risk          HITL
              │            │            │
              └────────────┼────────────┘
                           ▼
                      Tool Calling
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
               Success           Failure
                  │                 │
                  ▼                 ▼
             Supervisor          Retry
                                      │
                                      ▼
                                   Error
                                      │
                                      ▼
                                  Recovery
                                      │
                            Retry / Re-route /
                            Human Review
                                      │
                                      ▼
                                  Supervisor
                                      │
                                      ▼
                                  Finalizer
                                      │
                                      ▼
                                     END
```

---

# 19. Summary

Enterprise-Agent 的核心不是构建更多 Agent，而是构建一个能够对 Agent 行为进行**控制、持久化、恢复和治理**的 Runtime。

核心执行链路：

```text
Runtime Context
      ↓
Memory
      ↓
Supervisor
      ↓
Specialist
      ↓
Tool Governance
      ↓
Tool Execution
      ↓
Conditional Routing
      ↓
Recovery
      ↓
Checkpoint / Resume
      ↓
Finalizer
```

最终形成：

> **一个面向企业复杂任务执行的可治理、状态化、可恢复 Multi-Agent Runtime。**
