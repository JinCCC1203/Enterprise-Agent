#定义工具的元数据信息，与tool schema不同
from __future__ import annotations

from dataclasses import dataclass,field
from enum import Enum
from typing import FrozenSet

class ToolSource(str,Enum):
    """Tool 来源"""
    LOCAL="local"
    MCP="mcp"


class ToolRiskLevel(str,Enum):
    """Tool风险等级"""
    LOW="low"
    MEDIUM="medium"
    HIGH="high"

@dataclass(frozen=True, slots=True)
class ToolMetadata:
    """
    Runtime 层面的 Tool 元数据

    ToolMetadata 负责描述：Runtime 应该如何管理这个 Tool
    """
    name: str
    #工具类别
    category: str
    #用于基于规则的Selector 匹配
    tags: FrozenSet[str]=field(default_factory=frozenset)
    #当前工具允许哪些角色使用
    permissions: FrozenSet[str]=field(default_factory=frozenset)
    #工具来源
    source: ToolSource=ToolSource.LOCAL
    #风险等级，为后续 Tool Governance / HITL 做准备
    risk_level: ToolRiskLevel=ToolRiskLevel.LOW
    #可选描述，不取代 LangChain Tool 自身 description
    description: str=""

    def is_allowed_for(self,role: str) -> bool:
        """
        判断某个角色是否可以使用该Tool
        """
        return role in self.permissions

    def has_tag(self,tag: str) -> bool:
        """
        判断tool是否包含指定tag
        """
        return tag in self.tags



