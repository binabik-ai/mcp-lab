"""
Multi-provider LLM service for MCP Lab - Simplified Agent Version
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
    """Service for managing multi-provider LLM interactions with agent loop"""
    
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
    
    async def _execute_tool(self, tool_call: dict, tools_map: dict) -> dict:
        """Execute a single tool"""
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args', {})
        tool_start = time.time()
        
        if tool_name not in tools_map:
            error_msg = f"Tool '{tool_name}' not found. Available: {', '.join(tools_map.keys())}"
            logger.warning(error_msg)
            return {
                'tool_call_id': tool_call.get('id', tool_name),
                'name': tool_name,
                'args': tool_args,
                'result': error_msg,
                'error': True,
                'latency': 0,
                'iteration': 0
            }
        
        tool = tools_map[tool_name]
        try:
            if hasattr(tool, 'coroutine'):
                result = await tool.coroutine(**tool_args)
            else:
                result = await asyncio.to_thread(tool.func, **tool_args)
            
            return {
                'tool_call_id': tool_call.get('id', tool_name),
                'name': tool_name,
                'args': tool_args,
                'result': str(result)[:500],  # Truncate for storage
                'full_result': str(result),
                'error': False,
                'latency': time.time() - tool_start,
                'iteration': 0
            }
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {
                'tool_call_id': tool_call.get('id', tool_name),
                'name': tool_name,
                'args': tool_args,
                'result': f"Error: {str(e)}",
                'error': True,
                'latency': time.time() - tool_start,
                'iteration': 0
            }
    
    async def chat_with_tools(self, 
                              message: str,
                              session_id: str,
                              tools: list,
                              model: str,
                              provider: str,
                              debug_mode: bool = False) -> Dict[str, Any]:
        """Process chat with MCP tools support using agent loop"""
        start_time = time.time()
        session = self._get_or_create_session(session_id, model, provider)
        
        # Prepare messages - filter out empty messages
        system_prompt = "You are a helpful assistant with access to various tools through MCP servers. Use them when needed to help the user."
        messages = [SystemMessage(content=system_prompt)]
        
        # Add history but filter empty messages (important for Anthropic)
        for msg in session['memory'].messages:
            if hasattr(msg, 'content') and msg.content and str(msg.content).strip():
                messages.append(msg)
        
        messages.append(HumanMessage(content=message))
        
        # Track metrics
        all_tool_calls = []
        iterations = 0
        max_iterations = 5
        final_response = ""
        
        try:
            # Bind tools to LLM
            if tools:
                # Don't specify tool_choice for Groq
                if provider == 'groq':
                    llm_with_tools = session['llm'].bind_tools(tools)
                else:
                    llm_with_tools = session['llm'].bind_tools(tools, tool_choice="auto")
            else:
                llm_with_tools = session['llm']
            
            # Create tools map
            tools_map = {tool.name: tool for tool in tools}
            
            # Agent loop
            while iterations < max_iterations:
                iterations += 1
                
                # Call LLM
                try:
                    response = await asyncio.to_thread(llm_with_tools.invoke, messages)
                except Exception as e:
                    error_str = str(e)
                    # Handle Groq's tool choice error
                    if "Tool choice is none" in error_str and provider == 'groq':
                        logger.warning("Groq tool choice error, trying without tools")
                        try:
                            response = await asyncio.to_thread(session['llm'].invoke, messages)
                        except:
                            # If still fails, use the last response or a default
                            if final_response:
                                break
                            else:
                                final_response = "I encountered an error processing your request."
                                break
                    else:
                        raise
                
                # Check if response has content
                if hasattr(response, 'content') and response.content:
                    # Handle different response formats
                    content = response.content
                    
                    # Handle Anthropic's content blocks format
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict):
                                if block.get('type') == 'text':
                                    text_parts.append(block.get('text', ''))
                            elif isinstance(block, str):
                                text_parts.append(block)
                        final_response = ' '.join(text_parts).strip()
                    elif isinstance(content, str):
                        final_response = content
                    else:
                        final_response = str(content)
                
                # Check for tool calls
                if not hasattr(response, 'tool_calls') or not response.tool_calls:
                    # No more tool calls, we're done
                    break
                
                # Add the assistant's message to history
                messages.append(response)
                
                # Execute tools (in parallel if multiple)
                tool_calls = response.tool_calls
                
                if len(tool_calls) > 1:
                    # Execute multiple tools in parallel
                    tool_results = await asyncio.gather(
                        *[self._execute_tool(tc, tools_map) for tc in tool_calls]
                    )
                else:
                    # Execute single tool
                    tool_results = [await self._execute_tool(tool_calls[0], tools_map)]
                
                # Update iteration number in results
                for result in tool_results:
                    result['iteration'] = iterations
                    all_tool_calls.append(result)
                
                # Add tool results to messages
                for result in tool_results:
                    messages.append(ToolMessage(
                        content=result.get('full_result', result['result']),
                        tool_call_id=result['tool_call_id']
                    ))
            
            # Make sure we have a final response
            if not final_response and iterations > 0:
                # Try to get a final summary
                try:
                    final_msg = await asyncio.to_thread(session['llm'].invoke, messages)
                    if hasattr(final_msg, 'content') and final_msg.content:
                        content = final_msg.content
                        # Handle Anthropic's content blocks
                        if isinstance(content, list):
                            text_parts = []
                            for block in content:
                                if isinstance(block, dict):
                                    if block.get('type') == 'text':
                                        text_parts.append(block.get('text', ''))
                                elif isinstance(block, str):
                                    text_parts.append(block)
                            final_response = ' '.join(text_parts).strip()
                        else:
                            final_response = str(content)
                except:
                    final_response = "I've completed the requested operations. Please see the results above."
            
            # Final cleanup - ensure response is a clean string
            if isinstance(final_response, list):
                text_parts = []
                for item in final_response:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                        else:
                            # Skip tool_use blocks
                            continue
                    else:
                        text_parts.append(str(item))
                final_response = ' '.join(text_parts).strip()
            
            # Make sure final_response is a string
            if not isinstance(final_response, str):
                final_response = str(final_response)
            
            # Calculate metrics
            total_latency = time.time() - start_time
            total_tokens = len(message.split()) + len(str(final_response).split()) * 2
            
            # Save to memory (clean response for history)
            session['memory'].add_user_message(message)
            if final_response:
                session['memory'].add_ai_message(str(final_response))
            
            # Save to database
            if self.db_service:
                self.db_service.save_conversation(
                    session_id=session_id,
                    provider=provider,
                    model=model,
                    user_message=message,
                    ai_response=str(final_response),
                    total_tokens=total_tokens,
                    total_latency=total_latency,
                    tool_iterations=iterations,
                    tools_called_count=len(all_tool_calls),
                    tool_calls=[{k: v for k, v in tc.items() if k != 'full_result'} 
                               for tc in all_tool_calls]
                )
            
            return {
                'success': True,
                'response': final_response,
                'metrics': {
                    'tokens': total_tokens,
                    'latency': round(total_latency, 2),
                    'iterations': iterations,
                    'tools_called': len(all_tool_calls)
                },
                'tool_calls': all_tool_calls if debug_mode else None
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