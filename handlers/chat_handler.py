"""
Chat handler for MCP Lab
"""
from typing import Optional, List, Dict, Any
import logging
import re
from pathlib import Path
import shutil
import hashlib

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
            
            # Extract and process media from response
            response_text = result.get('response', '')
            # Ensure response is a string
            if not isinstance(response_text, str):
                response_text = str(response_text) if response_text else ''
            media_outputs = await self._extract_and_process_media(response_text)
            
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
    
    async def _extract_and_process_media(self, text: str) -> List[Dict[str, str]]:
        """Extract media URLs from response text and process local files"""
        media = []
        
        # Log the text being searched (truncated for readability)
        logger.info(f"Searching for media files in text (first 500 chars): {text[:500]}")
        
        # Extract file:// URLs and absolute paths to media files
        # Pattern 1: file:// URLs
        file_url_pattern = r'file://(/[^\s]+\.(?:html|png|jpg|jpeg|gif|svg))'
        # Pattern 2: Absolute paths without file:// prefix - more permissive
        # Look for paths that start with /Users or /home or /tmp or /var etc
        abs_path_pattern = r'(/(?:Users|home|tmp|var|opt|mnt)[^\s\'"<>]*\.(?:html|png|jpg|jpeg|gif|svg))\b'
        
        # Collect all file paths
        file_paths = []
        
        # Extract file:// URLs
        for match in re.finditer(file_url_pattern, text, re.IGNORECASE):
            file_paths.append(match.group(1))
            logger.info(f"Found file:// URL: {match.group(1)}")
        
        # Extract absolute paths
        for match in re.finditer(abs_path_pattern, text, re.IGNORECASE):
            path = match.group(1)
            # Avoid duplicates
            if path not in file_paths:
                file_paths.append(path)
                logger.info(f"Found absolute path: {path}")
        
        if file_paths:
            logger.info(f"Found {len(file_paths)} media files to process")
        
        # Process each file path
        for file_path in file_paths:
            
            # Process local file directly
            try:
                source_path = Path(file_path)
                if source_path.exists():
                    # Generate a unique filename based on content hash
                    with open(source_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()[:8]
                    
                    # Preserve original extension
                    ext = source_path.suffix
                    new_filename = f"{source_path.stem}_{file_hash}{ext}"
                    
                    # Copy to temp outputs directory
                    temp_outputs_dir = Path("temp_outputs")
                    temp_outputs_dir.mkdir(exist_ok=True)
                    dest_path = temp_outputs_dir / new_filename
                    shutil.copy2(source_path, dest_path)
                    
                    # Create the served URL
                    served_url = f"/outputs/{new_filename}"
                    
                    if '.html' in file_path.lower():
                        media.append({'type': 'iframe', 'url': served_url})
                    else:
                        media.append({'type': 'image', 'url': served_url})
                    
                    logger.info(f"Processed media file: {file_path} -> {served_url}")
                else:
                    logger.warning(f"File not found: {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
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