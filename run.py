#!/usr/bin/env python3
"""
MCP Lab - Application Launcher
"""
import os
import sys
from pathlib import Path
import uvicorn

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == '__main__':
    # Create necessary directories
    Path("config").mkdir(exist_ok=True)
    Path("frontend").mkdir(exist_ok=True)
    Path("services").mkdir(exist_ok=True)
    Path("handlers").mkdir(exist_ok=True)
    
    # Create default MCP config if it doesn't exist
    mcp_config_file = Path("config/mcp_config.json")
    if not mcp_config_file.exists():
        import json
        default_config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    "env": {}
                }
            }
        }
        mcp_config_file.write_text(json.dumps(default_config, indent=2))
        print(f"Created default MCP config at {mcp_config_file}")
    
    # Get configuration
    from config import Config
    
    print(f"""
╔══════════════════════════════════════╗
║          🧪 MCP Lab v1.0.0           ║
║   Test MCP servers with LLM providers║
╚══════════════════════════════════════╝

Starting server at http://{Config.HOST}:{Config.PORT}
Analytics at http://{Config.HOST}:{Config.PORT}/analytics

Press Ctrl+C to stop
""")
    
    # Run the application
    uvicorn.run(
        "app:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
        log_level="info" if not Config.DEBUG else "debug"
    )