"""
Multi-provider LLM service for MCP Lab
"""
from typing import Optional, List, Dict, Any
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.messages import ToolMessage
import asyncio
import time
import logging
import json

logger = logging.getLogger(__name__)

class LLMService:
    """Service for managing multi-provider LLM interactions"""
    
    def __init__(self, config, db_service):
        self.config = config
        self.db_service = db_service
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Provider API keys
        self.providers = {
            'groq': config.GROQ_API_KEY,
            'openai': config.OPENAI_API_KEY,
            'anthropic': config.ANTHROPIC_API_KEY
        }
        
        # Available models per provider
        self.models = {
            'groq': [
                "moonshotai/kimi-k2-instruct",
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
                "mixtral-8x7b-32768",
                "llama3-70b-8192",
                "llama3-8b-8192",
                "gemma-7b-it"
            ],
            'openai': [
                "gpt-4.1-mini",
                "gpt-4.1-nano",
                "gpt-5-mini",
                "gpt-5-nano",
                "gpt-4o-mini"
            ],
            'anthropic': [
                "claude-3-7-sonnet-20250219",
                "claude-sonnet-4-20250514",
                "claude-opus-4-1-20250805"
            ]
        }
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models for configured providers"""
        available = {}
        for provider, api_key in self.providers.items():
            if api_key:
                available[provider] = self.models.get(provider, [])
        return available
    
    def _create_llm(self, model: str, provider: str) -> Any:
        """Create provider-specific LLM instance"""
        if not self.providers.get(provider):
            raise ValueError(f"{provider} API key not configured")
        
        base_params = {
            'temperature': self.config.TEMPERATURE,
            'max_tokens': self.config.MAX_TOKENS
        }
        
        if provider == 'groq':
            return ChatGroq(
                groq_api_key=self.providers['groq'],
                model_name=model,
                **base_params
            )
        elif provider == 'openai':
            return ChatOpenAI(
                openai_api_key=self.providers['openai'],
                model_name=model,
                **base_params
            )
        elif provider == 'anthropic':
            return ChatAnthropic(
                anthropic_api_key=self.providers['anthropic'],
                model=model,
                **base_params
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _get_or_create_session(self, session_id: str, model: str, provider: str) -> Dict[str, Any]:
        """Get or create a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'model': model,
                'provider': provider,
                'memory': ChatMessageHistory(),
                'llm': self._create_llm(model, provider)
            }
        return self.sessions[session_id]
    
    async def chat_with_tools(self, 
                              message: str,
                              session_id: str,
                              tools: list,
                              model: str,
                              provider: str,
                              debug_mode: bool = False) -> Dict[str, Any]:
        """Process chat with MCP tools support"""
        start_time = time.time()
        session = self._get_or_create_session(session_id, model, provider)
        
        # Prepare messages
        system_prompt = "You are a helpful assistant with access to various tools through MCP servers. Use them when needed to help the user."
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(session['memory'].messages)
        messages.append(HumanMessage(content=message))
        
        # Track metrics
        tool_calls_details = []
        iterations = 0
        total_tokens = 0
        
        try:
            # Bind tools to LLM with explicit tool choice
            if tools:
                # Force tool choice to 'auto' to prevent the error
                llm_with_tools = session['llm'].bind_tools(tools, tool_choice="auto")
            else:
                llm_with_tools = session['llm']
            
            # Initial LLM call
            response = await asyncio.to_thread(llm_with_tools.invoke, messages)
            iterations = 1
            
            # Extract tool calls
            tool_calls = []
            if hasattr(response, 'tool_calls'):
                tool_calls = response.tool_calls
            
            # Process tool calls (simplified for now - can be enhanced)
            final_response = response.content if response.content else ""
            
            if tool_calls:
                # Execute tools
                tools_map = {tool.name: tool for tool in tools}
                for tool_call in tool_calls:
                    tool_name = tool_call.get('name')
                    tool_args = tool_call.get('args', {})
                    
                    if tool_name in tools_map:
                        tool = tools_map[tool_name]
                        tool_start = time.time()
                        
                        try:
                            if hasattr(tool, 'coroutine'):
                                result = await tool.coroutine(**tool_args)
                            else:
                                result = await asyncio.to_thread(tool.func, **tool_args)
                            
                            tool_latency = time.time() - tool_start
                            tool_calls_details.append({
                                'name': tool_name,
                                'args': tool_args,
                                'result': str(result)[:500],  # Truncate for storage
                                'latency': tool_latency,
                                'iteration': iterations
                            })
                            
                            # Add tool result to messages
                            messages.append(response)
                            messages.append(ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call.get('id', tool_name)
                            ))
                            
                        except Exception as e:
                            logger.error(f"Tool {tool_name} failed: {e}")
                            tool_calls_details.append({
                                'name': tool_name,
                                'args': tool_args,
                                'error': str(e),
                                'latency': time.time() - tool_start,
                                'iteration': iterations
                            })
                
                # Get final response after tool execution
                if tool_calls_details:
                    iterations += 1
                    final_response_obj = await asyncio.to_thread(session['llm'].invoke, messages)
                    final_response = final_response_obj.content
            
            # Calculate metrics
            total_latency = time.time() - start_time
            
            # Estimate tokens (rough calculation)
            if final_response:
                total_tokens = len(message.split()) + len(final_response.split()) * 2
            else:
                total_tokens = len(message.split()) * 2
            
            # Save to memory
            session['memory'].add_user_message(message)
            session['memory'].add_ai_message(final_response)
            
            # Save to database
            self.db_service.save_conversation(
                session_id=session_id,
                provider=provider,
                model=model,
                user_message=message,
                ai_response=final_response,
                total_tokens=total_tokens,
                total_latency=total_latency,
                tool_iterations=iterations,
                tools_called_count=len(tool_calls_details),
                tool_calls=tool_calls_details
            )
            
            return {
                'success': True,
                'response': final_response,
                'metrics': {
                    'tokens': total_tokens,
                    'latency': round(total_latency, 2),
                    'iterations': iterations,
                    'tools_called': len(tool_calls_details)
                },
                'tool_calls': tool_calls_details if debug_mode else None
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'response': f"Error: {str(e)}"
            }
    
    async def change_model(self, session_id: str, model: str, provider: str, keep_context: bool) -> bool:
        """Change model/provider for a session"""
        try:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session['model'] = model
                session['provider'] = provider
                session['llm'] = self._create_llm(model, provider)
                
                if not keep_context:
                    session['memory'].clear()
            else:
                self._get_or_create_session(session_id, model, provider)
            
            return True
        except Exception as e:
            logger.error(f"Error changing model: {e}")
            return False
    
    def clear_session(self, session_id: str):
        """Clear session memory"""
        if session_id in self.sessions:
            self.sessions[session_id]['memory'].clear()
    
    def cleanup_session(self, session_id: str):
        """Remove session"""
        if session_id in self.sessions:
            del self.sessions[session_id]