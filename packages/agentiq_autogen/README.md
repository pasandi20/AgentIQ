# AgentIQ AutoGen Integration

This package provides integration between AgentIQ and Microsoft's AutoGen framework.

## Installation

```bash
uv pip install -e '.[autogen]'
```

## Usage

```python
from aiq.plugins.autogen import AutoGenAssistantAgent, AutoGenUserProxyAgent

# Create agents
assistant = AutoGenAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant",
    tools=[your_tools]
)

user_proxy = AutoGenUserProxyAgent(
    name="user",
    system_message="You are a user proxy",
    tools=[your_tools]
)

# Use the agents
response = await assistant.run(messages=[...])
```

## Features

- Integration with AutoGen's AssistantAgent and UserProxyAgent
- Tool wrapping for AutoGen compatibility
- Callback handling for monitoring and logging
- LLM integration with AutoGen's OpenAI wrapper

## Requirements

- Python >= 3.8
- AgentIQ Core
- pyautogen >= 0.2.0 