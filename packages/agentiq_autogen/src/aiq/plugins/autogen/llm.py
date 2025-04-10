# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union
from pydantic import Field
from autogen import OpenAIWrapper
from aiq.llm import BaseLLM, LLMMessage, register_llm
from aiq.data_models.function import FunctionBaseConfig

class AutoGenLLMConfig(FunctionBaseConfig, name="autogen_llm"):
    """Configuration for AutoGen LLM."""
    
    model_name: str = Field(description="The model name to use")
    temperature: float = Field(default=0.0, description="The temperature to use")
    max_tokens: Optional[int] = Field(default=None, description="The maximum number of tokens to generate")
    top_p: Optional[float] = Field(default=None, description="The top-p value to use")
    frequency_penalty: Optional[float] = Field(default=None, description="The frequency penalty to use")
    presence_penalty: Optional[float] = Field(default=None, description="The presence penalty to use")
    stop: Optional[List[str]] = Field(default=None, description="The stop sequences to use")

@register_llm(config_type=AutoGenLLMConfig)
class AutoGenLLM(BaseLLM):
    """AutoGen LLM implementation."""
    
    def __init__(self, config: AutoGenLLMConfig):
        """Initialize AutoGen LLM.
        
        Args:
            config: Configuration dictionary containing AutoGen settings
        """
        super().__init__(config)
        self._llm = OpenAIWrapper(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            stop=config.stop
        )
        
    async def generate(self, messages: List[LLMMessage]) -> LLMMessage:
        """Generate a response from the LLM."""
        response = await self._llm.generate(messages)
        return LLMMessage(
            role="assistant",
            content=response.content
        )
    
    async def stream(self, messages: List[LLMMessage]) -> AsyncGenerator[LLMMessage, None]:
        """Stream a response from the LLM."""
        async for chunk in self._llm.stream(messages):
            yield LLMMessage(
                role="assistant",
                content=chunk.content
            )

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in the text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return self._llm.get_token_count(text) 