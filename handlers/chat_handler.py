"""
Chat handler for MCP Lab
"""
from typing import Optional, List, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)

class ChatHandler:
    """Handles chat interactions with MCP tools support"""
    
    def __init__(self, llm_service, mcp_service, websocket_manager, db_service):
        self.llm_service = llm_service
        self.mcp_service = mcp_service
        self.websocket_manager = websocket_manager
        self.db_service = db_service
        self.mcp_tools_bridge = None  # Set by app.py
    
    async def handle_message(self, session_id: str, message: str,
                            model: str, provider: str, debug_mode: bool = False):
        """Handle a chat message with MCP tools"""
        try:
            # Send typing indicator
            await self.websocket_manager.send_to_session(session_id, {
                "type": "chat_start"
            })
            
            # Get available tools
            tools = []
            if self.mcp_tools_bridge:
                tools = await self.mcp_tools_bridge.discover_tools()
            
            # Process with LLM
            result = await self.llm_service.chat_with_tools(
                message=message,
                session_id=session_id,
                tools=tools,
                model=model,
                provider=provider,
                debug_mode=debug_mode
            )
            
            # Extract media from response
            media_outputs = self._extract_media(result.get('response', ''))
            
            # Send response
            await self.websocket_manager.send_to_session(session_id, {
                "type": "chat_complete",
                "response": result.get('response', ''),
                "metrics": result.get('metrics', {}),
                "tool_calls": result.get('tool_calls', []) if debug_mode else None,
                "media_outputs": media_outputs,
                "success": result.get('success', False)
            })
            
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            await self.websocket_manager.send_to_session(session_id, {
                "type": "error",
                "message": str(e)
            })
    
    def _extract_media(self, text: str) -> List[Dict[str, str]]:
        """Extract media URLs from response text"""
        media = []
        
        # Extract file:// URLs
        file_pattern = r'file://(/[^\s]+\.(?:html|png|jpg|jpeg|gif|svg))'
        for match in re.finditer(file_pattern, text, re.IGNORECASE):
            file_path = match.group(0)
            if '.html' in file_path.lower():
                media.append({'type': 'iframe', 'url': file_path})
            else:
                media.append({'type': 'image', 'url': file_path})
        
        # Extract http(s) URLs that might be plots
        url_pattern = r'https?://[^\s]+(?:plotly|chart|graph|viz)[^\s]*'
        for match in re.finditer(url_pattern, text, re.IGNORECASE):
            media.append({'type': 'iframe', 'url': match.group(0)})
        
        return media
    
    async def change_model(self, session_id: str, model: str, provider: str, keep_context: bool) -> bool:
        """Change model/provider for a session"""
        return await self.llm_service.change_model(session_id, model, provider, keep_context)
    
    def clear_session(self, session_id: str):
        """Clear session memory"""
        self.llm_service.clear_session(session_id)
    
    def cleanup_session(self, session_id: str):
        """Clean up session"""
        self.llm_service.cleanup_session(session_id)