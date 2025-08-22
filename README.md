# 🧪 MCP Lab

A focused testing environment for MCP (Model Context Protocol) servers with multiple LLM providers. Compare how different models handle tool calling, track metrics, and analyze performance.

## Features

- **Multi-Provider Support**: Test with Groq, OpenAI, and Anthropic models
- **MCP Server Integration**: Connect to multiple MCP servers simultaneously
- **Smart Media Rendering**: Automatically embed plots, images, and iframes from tool outputs
- **Detailed Metrics**: Track tokens, latency, tool calls, and iterations
- **Debug Mode**: Toggle detailed tool execution visibility
- **Analytics Dashboard**: Built-in database viewer with filtering and export
- **SQLite Storage**: All conversations saved with full metadata for analysis

## Quick Start

### 1. Clone or Create Project Structure

```bash
mkdir mcp-lab
cd mcp-lab

# Create directory structure
mkdir -p config services handlers frontend

# Copy all the provided files to their respective locations
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required: At least one LLM provider API key:
- `GROQ_API_KEY` - Get from [console.groq.com](https://console.groq.com)
- `OPENAI_API_KEY` - Get from [platform.openai.com](https://platform.openai.com)
- `ANTHROPIC_API_KEY` - Get from [console.anthropic.com](https://console.anthropic.com)

### 4. Configure MCP Servers

Edit `config/mcp_config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {}
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": {}
    },
    "puppeteer": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
      "env": {}
    }
  }
}
```

### 5. Run the Application

```bash
python run.py
```

Open your browser to:
- **Main Lab**: http://localhost:8000
- **Analytics**: http://localhost:8000/analytics

## Project Structure

```
mcp-lab/
├── app.py                  # FastAPI backend
├── config.py              # Configuration
├── run.py                 # Application launcher
├── requirements.txt       # Python dependencies
├── .env                   # Your API keys (create from .env.example)
├── mcp_lab.db            # SQLite database (created automatically)
│
├── config/
│   └── mcp_config.json   # MCP server configurations
│
├── services/
│   ├── llm_service.py    # Multi-provider LLM management
│   ├── mcp_service.py    # MCP server management
│   ├── mcp_tools_bridge.py # Tool conversion bridge
│   └── db_service.py     # Database operations
│
├── handlers/
│   ├── chat_handler.py   # Chat message processing
│   └── websocket_manager.py # WebSocket connections
│
└── frontend/
    ├── index.html        # Main chat interface
    ├── analytics.html    # Analytics dashboard
    └── style.css        # Styling
```

## Usage Guide

### Testing MCP Tools

1. **Start a conversation** - The AI will automatically discover available tools
2. **Toggle Debug Mode** - See detailed tool execution, arguments, and results
3. **Compare Providers** - Switch between providers/models to compare behavior

### Understanding Metrics

- **Tokens**: Approximate token count for the exchange
- **Latency**: Total time from request to response
- **Tools Called**: Number of MCP tool invocations
- **Iterations**: Number of reasoning cycles

### Media Rendering

The system automatically detects and renders:
- `file:///path/to/plot.html` → Embedded iframe
- Image files → Inline images
- Plotly/chart URLs → Embedded visualizations

### Analytics Dashboard

- View all conversations with filtering
- Compare provider/model performance
- Export data as JSON or CSV
- Track average metrics per provider

## Adding MCP Servers

1. Install the MCP server:
```bash
npm install -g @modelcontextprotocol/server-name
```

2. Add to `config/mcp_config.json`:
```json
{
  "mcpServers": {
    "your-server": {
      "command": "path-to-executable",
      "args": ["arg1", "arg2"],
      "env": {
        "ENV_VAR": "value"
      }
    }
  }
}
```

3. Restart MCP Lab

## Missing Files

You'll need to copy these files from your original project or recreate them:

### `services/mcp_service.py`
Copy your existing MCP service file as-is (it's already well-designed for multiple servers)

### `services/mcp_tools_bridge.py`
Copy your existing MCP tools bridge file as-is

### `handlers/websocket_manager.py`
Create a simple WebSocket manager:

```python
from typing import Dict
from fastapi import WebSocket
import json
import logging

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")
    
    async def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_to_session(self, session_id: str, data: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(data)
            except Exception as e:
                logger.error(f"Error sending to session {session_id}: {e}")
                await self.disconnect(session_id)
```

## Troubleshooting

### MCP Servers Not Connecting
- Check that Node.js and npm are installed
- Verify MCP server packages are installed
- Check the command paths in `mcp_config.json`
- Look at console logs for specific error messages

### No Models Available
- Verify at least one API key is set in `.env`
- Check that the API key is valid
- Ensure you have access to the models

### Database Issues
- Delete `mcp_lab.db` to start fresh
- Check write permissions in the directory

## Development

### Adding New Features

1. **New Tool Types**: Edit `handlers/chat_handler.py` `_extract_media()` method
2. **New Metrics**: Add to database schema in `services/db_service.py`
3. **New Providers**: Add to `services/llm_service.py` models and provider setup

### Debug Mode

Set `DEBUG=True` in `.env` for:
- Auto-reload on code changes
- Detailed logging
- Error stack traces

## License

MIT

## Contributing

Contributions welcome! This is a testing tool for MCP development, so features that help with debugging and analysis are especially appreciated.

## Acknowledgments

Built for testing MCP (Model Context Protocol) servers with various LLM providers. MCP is developed by Anthropic.