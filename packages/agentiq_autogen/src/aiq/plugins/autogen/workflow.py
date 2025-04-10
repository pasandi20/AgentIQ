from typing import Any, Dict, List, Optional
from pydantic import Field
from autogen import GroupChat, GroupChatManager, AssistantAgent
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.component_ref import FunctionRef, LLMRef
import logging

logger = logging.getLogger(__name__)

class AutoGenGroupChatWorkflowConfig(FunctionBaseConfig, name="autogen_group_chat"):
    """Configuration for AutoGen Group Chat workflow."""
    
    agents: List[Dict[str, Any]] = Field(description="List of agent configurations")
    tool_names: List[FunctionRef] = Field(default_factory=list, description="List of tools available to the agents")
    llm_name: LLMRef = Field(description="The LLM to use for the agents")
    verbose: bool = Field(default=False, description="Set verbosity of logging")
    retry_parsing_errors: bool = Field(default=True, description="Whether to retry on parsing errors")
    max_retries: int = Field(default=3, description="Maximum number of retries")

@register_function(config_type=AutoGenGroupChatWorkflowConfig)
async def autogen_group_chat_workflow(config: AutoGenGroupChatWorkflowConfig, builder: Any):
    """AutoGen Group Chat workflow implementation."""
    
    async def _response_fn(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info("Starting AutoGen Group Chat workflow")
        # Create AutoGen agents
        autogen_agents = []
        for agent_config in config.agents:
            logger.info(f"Creating agent: {agent_config['name']}")
            agent = AssistantAgent(
                name=agent_config["name"],
                system_message=agent_config["role"],
                **agent_config.get("kwargs", {})
            )
            autogen_agents.append(agent)
            
        # Create group chat
        logger.info("Creating group chat")
        group_chat = GroupChat(
            agents=autogen_agents,
            messages=messages,
            **config.model_dump(exclude_none=True)
        )
        
        # Create manager
        logger.info("Creating group chat manager")
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=config.model_dump(exclude_none=True)
        )
        
        # Run the chat
        logger.info("Running group chat")
        response = await manager.run()
        return {"content": response}

    try:
        yield _response_fn
    except GeneratorExit:
        logger.warning("Workflow exited early!")
    finally:
        logger.info("Cleaning up AutoGen Group Chat workflow.") 