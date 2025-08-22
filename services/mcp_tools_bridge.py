"""
Bridge between MCP servers and LangChain tools
"""
from typing import List, Dict, Any, Optional, Type
from langchain.tools import Tool, StructuredTool
try:
    # Try pydantic v2 first
    from pydantic import BaseModel as BaseModelV1, Field, create_model
    create_model_v1 = create_model
except ImportError:
    # Fall back to v1 compatibility
    from pydantic.v1 import BaseModel as BaseModelV1, Field, create_model as create_model_v1
import json
import logging
from datetime import datetime, date
from decimal import Decimal

logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

class MCPToolsBridge:
    """Converts MCP tools to LangChain tools"""
    
    def __init__(self, mcp_service):
        self.mcp_service = mcp_service
        self.tools = []
        self.tool_map = {}  # Maps tool names to (server, original_name)
    
    def _create_pydantic_model(self, tool_name: str, input_schema: dict) -> Type[BaseModelV1]:
        """Create a Pydantic model from MCP input schema"""
        properties = input_schema.get('properties', {})
        required = input_schema.get('required', [])
        
        field_definitions = {}
        for prop_name, prop_schema in properties.items():
            # Determine field type
            prop_type = prop_schema.get('type', 'string')
            type_mapping = {
                'string': str,
                'number': float,
                'integer': int,
                'boolean': bool,
                'array': list,
                'object': dict
            }
            field_type = type_mapping.get(prop_type, str)
            
            description = prop_schema.get('description', f'{prop_name} parameter')
            
            if prop_name in required:
                field_definitions[prop_name] = (field_type, Field(description=description))
            else:
                field_definitions[prop_name] = (Optional[field_type], Field(default=None, description=description))
        
        return create_model_v1(f'{tool_name}_args', **field_definitions)
    
    async def discover_tools(self) -> List[Tool]:
        """Discover and create LangChain tools from all MCP servers"""
        self.tools = []
        self.tool_map = {}
        
        all_mcp_tools = await self.mcp_service.get_available_tools()
        
        for tool_info in all_mcp_tools:
            server_name = tool_info['server']
            tool_name = tool_info['name']
            
            # Create unique name if multiple servers have same tool
            unique_name = f"{server_name}_{tool_name}" if len(self.mcp_service.servers) > 1 else tool_name
            self.tool_map[unique_name] = (server_name, tool_name)
            
            description = tool_info.get('description', f"Tool {tool_name} from {server_name}")
            input_schema = tool_info.get('inputSchema', {})
            
            # Create tool based on schema complexity
            if input_schema and 'properties' in input_schema and input_schema['properties']:
                # Structured tool with Pydantic model
                args_model = self._create_pydantic_model(unique_name, input_schema)
                
                tool = StructuredTool(
                    name=unique_name,
                    description=description,
                    func=lambda **kwargs: "This tool requires async execution",
                    coroutine=self._create_tool_async_func(server_name, tool_name),
                    args_schema=args_model
                )
            else:
                # Simple tool without structured args
                tool = Tool(
                    name=unique_name,
                    description=description,
                    func=lambda **kwargs: "This tool requires async execution",
                    coroutine=self._create_tool_async_func(server_name, tool_name)
                )
            
            self.tools.append(tool)
            logger.info(f"Registered tool: {unique_name}")
        
        return self.tools
    
    def _create_tool_async_func(self, server: str, tool_name: str):
        """Create an async function for tool execution with proper closure"""
        # Capture server and tool_name in the closure properly
        async def tool_func(**kwargs):
            try:
                # Clean up arguments - remove None values and strip strings
                kwargs = {
                    k: (v.strip() if isinstance(v, str) else v) 
                    for k, v in kwargs.items() 
                    if v is not None
                }
                
                logger.info(f"EXECUTING MCP TOOL: {server}.{tool_name}")
                logger.info(f"  Arguments: {kwargs}")
                
                result = await self.mcp_service.call_tool(server, tool_name, kwargs)
                
                logger.info(f"  Result: {str(result)[:200]}...")
                
                # Handle different result types
                if isinstance(result, dict):
                    return json.dumps(result, indent=2, cls=DateTimeEncoder)
                elif isinstance(result, (list, tuple)):
                    return json.dumps(result, indent=2, cls=DateTimeEncoder)
                else:
                    return str(result)
                    
            except Exception as e:
                error_msg = f"Error calling {tool_name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return error_msg
        
        return tool_func