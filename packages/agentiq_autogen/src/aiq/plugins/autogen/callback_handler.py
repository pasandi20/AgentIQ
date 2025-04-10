# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional
from autogen import Agent
from aiq.core.callback import BaseCallback, register_callback

@register_callback("autogen")
class AutoGenCallback(BaseCallback):
    """Callback handler for AutoGen events."""
    
    def __init__(self):
        """Initialize AutoGen callback handler."""
        super().__init__()
        
    async def on_llm_start(
        self,
        agent: Agent,
        messages: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Called when LLM starts processing.
        
        Args:
            agent: The AutoGen agent
            messages: The messages being processed
            **kwargs: Additional arguments
        """
        await self.handle_event(
            "llm_start",
            {
                "agent_name": agent.name,
                "messages": messages
            }
        )
        
    async def on_llm_end(
        self,
        agent: Agent,
        response: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Called when LLM completes processing.
        
        Args:
            agent: The AutoGen agent
            response: The response from the LLM
            **kwargs: Additional arguments
        """
        await self.handle_event(
            "llm_end",
            {
                "agent_name": agent.name,
                "response": response
            }
        )
        
    async def on_tool_start(
        self,
        agent: Agent,
        tool_name: str,
        tool_input: Any,
        **kwargs: Any
    ) -> None:
        """Called when a tool starts execution.
        
        Args:
            agent: The AutoGen agent
            tool_name: Name of the tool being used
            tool_input: Input provided to the tool
            **kwargs: Additional arguments
        """
        await self.handle_event(
            "tool_start",
            {
                "agent_name": agent.name,
                "tool_name": tool_name,
                "tool_input": tool_input
            }
        )
        
    async def on_tool_end(
        self,
        agent: Agent,
        tool_name: str,
        tool_output: Any,
        **kwargs: Any
    ) -> None:
        """Called when a tool completes execution.
        
        Args:
            agent: The AutoGen agent
            tool_name: Name of the tool that was used
            tool_output: Output from the tool
            **kwargs: Additional arguments
        """
        await self.handle_event(
            "tool_end",
            {
                "agent_name": agent.name,
                "tool_name": tool_name,
                "tool_output": tool_output
            }
        )
        
    async def on_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """Called when an error occurs.
        
        Args:
            error: The exception that occurred
            **kwargs: Additional arguments
        """
        await self.handle_event(
            "error",
            {
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
        ) 