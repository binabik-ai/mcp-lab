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
        
        # Prepare messages - filter out empty messages for Anthropic
        system_prompt = "You are a helpful assistant with access to various tools through MCP servers. Use them when needed to help the user."
        messages = [SystemMessage(content=system_prompt)]
        
        # Add history but filter empty messages
        for msg in session['memory'].messages:
            if hasattr(msg, 'content') and msg.content and str(msg.content).strip():
                messages.append(msg)
        
        messages.append(HumanMessage(content=message))
        
        # Track metrics
        tool_calls_details = []
        iterations = 0
        total_tokens = 0
        
        try:
            # Bind tools to LLM
            if tools:
                # Don't specify tool_choice for Groq - it handles this automatically
                if provider == 'groq':
                    llm_with_tools = session['llm'].bind_tools(tools)
                else:
                    llm_with_tools = session['llm'].bind_tools(tools, tool_choice="auto")
            else:
                llm_with_tools = session['llm']
            
            # Create tools map for easy lookup
            tools_map = {tool.name: tool for tool in tools}
            
            # Agent loop - up to 5 iterations
            max_iterations = 5
            final_response = ""
            
            while iterations < max_iterations:
                iterations += 1
                
                # Call LLM
                try:
                    response = await asyncio.to_thread(llm_with_tools.invoke, messages)
                except Exception as e:
                    # Handle Groq's tool choice error
                    if "Tool choice is none" in str(e) and provider == 'groq':
                        logger.warning("Groq tool choice error, trying without tools")
                        response = await asyncio.to_thread(session['llm'].invoke, messages)
                    else:
                        raise
                
                # Check if response has content
                if hasattr(response, 'content') and response.content:
                    final_response = response.content
                
                # Check for tool calls
                if not hasattr(response, 'tool_calls') or not response.tool_calls:
                    # No more tool calls, we're done
                    break
                
                # Add the assistant's message with tool calls to history
                messages.append(response)
                
                # Execute tools in parallel if multiple
                tool_calls = response.tool_calls
            
                # Execute tools (potentially in parallel)
                tool_results = []
                
                if len(tool_calls) > 1:
                    # Execute multiple tools in parallel
                    import asyncio
                    
                    async def execute_tool(tool_call):
                        tool_name = tool_call.get('name')
                        tool_args = tool_call.get('args', {})
                        tool_start = time.time()
                        
                        if tool_name not in tools_map:
                            error_msg = f"Tool '{tool_name}' not found"
                            logger.warning(error_msg)
                            return {
                                'tool_call_id': tool_call.get('id', tool_name),
                                'result': error_msg,
                                'error': True,
                                'tool_call': tool_call,
                                'latency': 0
                            }
                        
                        tool = tools_map[tool_name]
                        try:
                            if hasattr(tool, 'coroutine'):
                                result = await tool.coroutine(**tool_args)
                            else:
                                result = await asyncio.to_thread(tool.func, **tool_args)
                            
                            return {
                                'tool_call_id': tool_call.get('id', tool_name),
                                'result': str(result),
                                'tool_call': tool_call,
                                'latency': time.time() - tool_start
                            }
                        except Exception as e:
                            logger.error(f"Tool {tool_name} failed: {e}")
                            return {
                                'tool_call_id': tool_call.get('id', tool_name),
                                'result': f"Error: {str(e)}",
                                'error': True,
                                'tool_call': tool_call,
                                'latency': time.time() - tool_start
                            }
                    
                    # Execute all tools in parallel
                    tool_results = await asyncio.gather(*[execute_tool(tc) for tc in tool_calls])
                    
                else:
                    # Execute single tool
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
                    else:
                        # Tool not found - add message for LLM to retry
                        error_msg = f"Tool '{tool_name}' not found. Available tools: {', '.join(tools_map.keys())}"
                        tool_not_found_messages.append(error_msg)
                        tool_calls_details.append({
                            'name': tool_name,
                            'args': tool_args,
                            'error': error_msg,
                            'latency': 0,
                            'iteration': iterations
                        })
                        logger.warning(f"Tool not found: {tool_name}")
                
                # If there were tool not found errors, add them to messages and retry
                if tool_not_found_messages:
                    messages.append(response)
                    for error_msg in tool_not_found_messages:
                        messages.append(ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_name
                        ))
                    
                    # Retry with correct tools
                    iterations += 1
                    retry_response = await asyncio.to_thread(llm_with_tools.invoke, messages)
                    
                    # Process retry response
                    if hasattr(retry_response, 'tool_calls') and retry_response.tool_calls:
                        for tool_call in retry_response.tool_calls:
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
                                        'result': str(result)[:500],
                                        'latency': tool_latency,
                                        'iteration': iterations
                                    })
                                    
                                    messages.append(retry_response)
                                    messages.append(ToolMessage(
                                        content=str(result),
                                        tool_call_id=tool_call.get('id', tool_name)
                                    ))
                                    
                                except Exception as e:
                                    logger.error(f"Tool {tool_name} failed on retry: {e}")
                                    tool_calls_details.append({
                                        'name': tool_name,
                                        'args': tool_args,
                                        'error': str(e),
                                        'latency': time.time() - tool_start,
                                        'iteration': iterations
                                    })
                
                # Get final response after tool execution
                if tool_calls_details or tool_not_found_messages:
                    iterations += 1
                    try:
                        final_response_obj = await asyncio.to_thread(session['llm'].invoke, messages)
                        final_response = final_response_obj.content
                        
                        # Check if the model wants to call more tools
                        max_iterations = 5
                        while hasattr(final_response_obj, 'tool_calls') and final_response_obj.tool_calls and iterations < max_iterations:
                            iterations += 1
                            messages.append(final_response_obj)
                            
                            for tool_call in final_response_obj.tool_calls:
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
                                            'result': str(result)[:500],
                                            'latency': tool_latency,
                                            'iteration': iterations
                                        })
                                        
                                        messages.append(ToolMessage(
                                            content=str(result),
                                            tool_call_id=tool_call.get('id', tool_name)
                                        ))
                                    except Exception as e:
                                        logger.error(f"Tool {tool_name} failed: {e}")
                                        messages.append(ToolMessage(
                                            content=f"Error: {str(e)}",
                                            tool_call_id=tool_call.get('id', tool_name)
                                        ))
                            
                            # Get next response
                            try:
                                final_response_obj = await asyncio.to_thread(session['llm'].invoke, messages)
                                final_response = final_response_obj.content
                            except Exception as e:
                                # If we get a Groq error about tool choice, just use the last response
                                if "Tool choice is none" in str(e) and provider == 'groq':
                                    logger.warning("Groq tool choice error, using last valid response")
                                    break
                                else:
                                    raise
                                    
                    except Exception as e:
                        # If we get a Groq error about tool choice after tool execution, it's fine
                        if "Tool choice is none" in str(e) and provider == 'groq' and tool_calls_details:
                            logger.warning("Groq tool choice error after successful tool execution, continuing")
                            # Try to get a final response without tools
                            try:
                                final_response_obj = await asyncio.to_thread(session['llm'].invoke, messages)
                                final_response = final_response_obj.content
                            except:
                                # If that fails too, create a summary based on tool results
                                final_response = "I've executed the requested tools. Based on the results above, the operation was completed successfully."
                        else:
                            raise
            
            # Calculate metrics
            total_latency = time.time() - start_time
            
            # Handle case where final_response might be a list (Anthropic sometimes returns this)
            if isinstance(final_response, list):
                # Extract text content from list
                final_response = ' '.join(str(item) for item in final_response)
            
            # Estimate tokens (rough calculation)
            if final_response:
                total_tokens = len(message.split()) + len(str(final_response).split()) * 2
            else:
                total_tokens = len(message.split()) * 2
            
            # Save to memory
            session['memory'].add_user_message(message)
            session['memory'].add_ai_message(final_response)
            
            # Save to database
            if self.db_service:
                self.db_service.save_conversation(
                    session_id=session_id,
                    provider=provider,
                    model=model,
                    user_message=message,
                    ai_response=str(final_response),  # Ensure it's a string
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
            error_str = str(e)
            logger.error(f"Error in chat: {e}", exc_info=True)
            
            # Handle specific Groq error about tool_choice
            if "Tool choice is none, but model called a tool" in error_str and provider == 'groq':
                try:
                    # Extract the failed tool call from error message
                    import re
                    match = re.search(r'"name":\s*"([^"]+)"', error_str)
                    if match:
                        failed_tool_name = match.group(1)
                        
                        # Add helpful message to conversation
                        error_msg = f"I tried to call a tool '{failed_tool_name}' that doesn't exist. Let me check the available tools."
                        messages.append(AIMessage(content=error_msg))
                        
                        # List available tools
                        available_tools = [tool.name for tool in tools]
                        tools_msg = f"Available tools are: {', '.join(available_tools)}. Let me use the correct tool."
                        messages.append(SystemMessage(content=tools_msg))
                        
                        # Retry with explicit tool list reminder
                        retry_response = await asyncio.to_thread(llm_with_tools.invoke, messages)
                        
                        # Process the retry
                        if hasattr(retry_response, 'tool_calls') and retry_response.tool_calls:
                            for tool_call in retry_response.tool_calls:
                                tool_name = tool_call.get('name')
                                tool_args = tool_call.get('args', {})
                                
                                tools_map = {tool.name: tool for tool in tools}
                                if tool_name in tools_map:
                                    tool = tools_map[tool_name]
                                    
                                    try:
                                        if hasattr(tool, 'coroutine'):
                                            result = await tool.coroutine(**tool_args)
                                        else:
                                            result = await asyncio.to_thread(tool.func, **tool_args)
                                        
                                        messages.append(retry_response)
                                        messages.append(ToolMessage(
                                            content=str(result),
                                            tool_call_id=tool_call.get('id', tool_name)
                                        ))
                                        
                                        # Get final response
                                        final_response_obj = await asyncio.to_thread(session['llm'].invoke, messages)
                                        final_response = final_response_obj.content
                                        
                                        # Save to memory
                                        session['memory'].add_user_message(message)
                                        session['memory'].add_ai_message(final_response)
                                        
                                        return {
                                            'success': True,
                                            'response': final_response,
                                            'metrics': {
                                                'tokens': len(message.split()) + len(final_response.split()) * 2,
                                                'latency': round(time.time() - start_time, 2),
                                                'iterations': 2,
                                                'tools_called': 1
                                            },
                                            'tool_calls': [{
                                                'name': tool_name,
                                                'args': tool_args,
                                                'result': str(result)[:500],
                                                'latency': 0,
                                                'iteration': 2
                                            }] if debug_mode else None
                                        }
                                    except Exception as tool_error:
                                        logger.error(f"Tool execution failed: {tool_error}")
                        else:
                            final_response = retry_response.content if retry_response.content else "I apologize, but I couldn't find the right tool to use."
                            
                            # Save to memory
                            session['memory'].add_user_message(message)
                            session['memory'].add_ai_message(final_response)
                            
                            return {
                                'success': True,
                                'response': final_response,
                                'metrics': {
                                    'tokens': len(message.split()) + len(final_response.split()) * 2,
                                    'latency': round(time.time() - start_time, 2),
                                    'iterations': 2,
                                    'tools_called': 0
                                },
                                'tool_calls': None
                            }
                        
                except Exception as retry_error:
                    logger.error(f"Retry after Groq error failed: {retry_error}")
            
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