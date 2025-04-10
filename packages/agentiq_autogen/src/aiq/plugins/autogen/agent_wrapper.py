# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from aiq.core.agent import BaseAgent, register_agent
from aiq.core.tool import BaseTool

@register_agent("autogen_assistant")
class AutoGenAssistantAgent(BaseAgent):
    """AutoGen Assistant Agent wrapper."""
    
    def __init__(
        self,
        name: str,
        system_message: str,
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any
    ):
        """Initialize AutoGen Assistant Agent.
        
        Args:
            name: Name of the agent
            system_message: System message for the agent
            tools: Optional list of tools available to the agent
            **kwargs: Additional keyword arguments for AutoGen agent
        """
        super().__init__(name, system_message, tools)
        self.agent = AssistantAgent(
            name=name,
            system_message=system_message,
            **kwargs
        )
        
    async def run(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Run the agent on the given messages.
        
        Args:
            messages: List of messages to process
            **kwargs: Additional keyword arguments
            
        Returns:
            Agent's response
        """
        # Convert messages to AutoGen format if needed
        response = await self.agent.generate_response(
            messages=messages,
            **kwargs
        )
        return {"content": response}

@register_agent("autogen_user_proxy")
class AutoGenUserProxyAgent(BaseAgent):
    """AutoGen User Proxy Agent wrapper."""
    
    def __init__(
        self,
        name: str,
        system_message: str,
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any
    ):
        """Initialize AutoGen User Proxy Agent.
        
        Args:
            name: Name of the agent
            system_message: System message for the agent
            tools: Optional list of tools available to the agent
            **kwargs: Additional keyword arguments for AutoGen agent
        """
        super().__init__(name, system_message, tools)
        self.agent = UserProxyAgent(
            name=name,
            system_message=system_message,
            **kwargs
        )
        
    async def run(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Run the agent on the given messages.
        
        Args:
            messages: List of messages to process
            **kwargs: Additional keyword arguments
            
        Returns:
            Agent's response
        """
        response = await self.agent.generate_response(
            messages=messages,
            **kwargs
        )
        return {"content": response} 