"""
MCP Lab - FastAPI backend for MCP testing with multiple LLM providers
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import json
import asyncio
import uuid
from pathlib import Path
from datetime import datetime
import logging

from config import Config
from services.llm_service import LLMService
from services.mcp_service import MCPService
from services.db_service import DatabaseService
from handlers.chat_handler import ChatHandler
from handlers.websocket_manager import WebSocketManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
websocket_manager = WebSocketManager()
db_service = None
llm_service = None
mcp_service = None
chat_handler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global llm_service, mcp_service, chat_handler, db_service
    
    # Startup
    logger.info("Starting MCP Lab services...")
    
    # Initialize services
    db_service = DatabaseService()
    db_service.init_database()
    
    llm_service = LLMService(Config, db_service)
    mcp_service = MCPService(Config)
    
    # Start MCP servers
    await mcp_service.initialize_servers()
    
    # Create MCP tools bridge
    from services.mcp_tools_bridge import MCPToolsBridge
    mcp_tools_bridge = MCPToolsBridge(mcp_service)
    
    # Discover initial tools
    tools = await mcp_tools_bridge.discover_tools()
    logger.info(f"=" * 60)
    logger.info(f"MCP LAB: Connected to {len(mcp_service.servers)} servers")
    logger.info(f"DISCOVERED {len(tools)} MCP TOOLS")
    for tool in tools[:10]:  # Show first 10 tools
        logger.info(f"  🔧 {tool.name}: {tool.description[:60]}...")
    if len(tools) > 10:
        logger.info(f"  ... and {len(tools) - 10} more tools")
    logger.info(f"=" * 60)
    
    # Create chat handler with MCP tools bridge
    chat_handler = ChatHandler(llm_service, mcp_service, websocket_manager, db_service)
    chat_handler.mcp_tools_bridge = mcp_tools_bridge
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP Lab services...")
    await mcp_service.shutdown()


# Create FastAPI app
app = FastAPI(
    title="MCP Lab",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path("frontend")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# HTTP Routes
@app.get("/")
async def root():
    """Serve the main application"""
    return FileResponse("frontend/index.html")

@app.get("/analytics")
async def analytics():
    """Serve the analytics page"""
    return FileResponse("frontend/analytics.html")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "mcp-lab",
        "active_sessions": len(websocket_manager.active_connections),
        "mcp_servers": mcp_service.get_server_status() if mcp_service else {}
    }

@app.get("/api/models")
async def get_models():
    """Get available models across all providers"""
    return {"models": llm_service.get_available_models()}

@app.get("/api/mcp/status")
async def get_mcp_status():
    """Get MCP servers status and tool count"""
    servers = mcp_service.get_server_status()
    tool_count = 0
    
    if hasattr(chat_handler, 'mcp_tools_bridge') and chat_handler.mcp_tools_bridge:
        tool_count = len(chat_handler.mcp_tools_bridge.tools)
    
    return {
        "servers": servers,
        "total_tools": tool_count,
        "connected_count": sum(1 for s in servers.values() if s.get('state') == 'connected')
    }

@app.get("/api/analytics/conversations")
async def get_conversations(
    limit: int = 100,
    offset: int = 0,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get conversations with filtering"""
    filters = {}
    if provider:
        filters['provider'] = provider
    if model:
        filters['model'] = model
    if start_date:
        filters['start_date'] = start_date
    if end_date:
        filters['end_date'] = end_date
    
    conversations = db_service.get_conversations(limit, offset, filters)
    return {"conversations": conversations, "total": len(conversations)}

@app.get("/api/analytics/stats")
async def get_analytics_stats():
    """Get analytics statistics"""
    stats = db_service.get_stats()
    return stats

@app.delete("/api/analytics/clear")
async def clear_database():
    """Clear all conversation history"""
    db_service.clear_all()
    return {"status": "cleared"}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection handler"""
    session_id = str(uuid.uuid4())
    await websocket_manager.connect(websocket, session_id)
    
    try:
        # Send initial connection info
        mcp_status = await get_mcp_status()
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "mcp_status": mcp_status
        })
        
        # Listen for messages
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "chat_message":
                # Process chat message
                await chat_handler.handle_message(
                    session_id=session_id,
                    message=data.get("message"),
                    model=data.get("model"),
                    provider=data.get("provider"),
                    debug_mode=data.get("debug_mode", False)
                )
                
            elif message_type == "change_model":
                keep_context = data.get("keep_context", False)
                success = await chat_handler.change_model(
                    session_id, 
                    data.get("model"),
                    data.get("provider"),
                    keep_context
                )
                await websocket.send_json({
                    "type": "model_changed",
                    "success": success,
                    "model": data.get("model"),
                    "provider": data.get("provider"),
                    "context_kept": keep_context
                })
                
            elif message_type == "clear_history":
                chat_handler.clear_session(session_id)
                await websocket.send_json({
                    "type": "history_cleared"
                })
                
    except WebSocketDisconnect:
        logger.info(f"Client {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
    finally:
        await websocket_manager.disconnect(session_id)
        chat_handler.cleanup_session(session_id)