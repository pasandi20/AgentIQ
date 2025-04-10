# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, Optional
from autogen import Tool
from aiq.core.tool import BaseTool, register_tool_wrapper

@register_tool_wrapper("autogen")
class AutoGenToolWrapper:
    """Wrapper for converting AIQ tools to AutoGen tools."""
    
    @classmethod
    def wrap_tool(cls, tool: BaseTool) -> Tool:
        """Wrap an AIQ tool as an AutoGen tool.
        
        Args:
            tool: The AIQ tool to wrap
            
        Returns:
            AutoGen Tool instance
        """
        # Create function that handles the tool execution
        async def tool_func(*args: Any, **kwargs: Any) -> Any:
            result = await tool.run(*args, **kwargs)
            return result.get("output") if isinstance(result, dict) else result
            
        # Create AutoGen tool with the wrapped function
        return Tool(
            name=tool.name,
            description=tool.description,
            func=tool_func,
            async_fn=True
        )
    
    @classmethod
    def wrap_tools(cls, tools: List[BaseTool]) -> List[Tool]:
        """Wrap multiple AIQ tools as AutoGen tools.
        
        Args:
            tools: List of AIQ tools to wrap
            
        Returns:
            List of AutoGen Tool instances
        """
        return [cls.wrap_tool(tool) for tool in tools] 