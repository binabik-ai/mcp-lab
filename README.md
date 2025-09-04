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

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required: At least one LLM provider API key:
- `GROQ_API_KEY` - Get from [console.groq.com](https://console.groq.com)
- `OPENAI_API_KEY` - Get from [platform.openai.com](https://platform.openai.com)
- `ANTHROPIC_API_KEY` - Get from [console.anthropic.com](https://console.anthropic.com)

### 3. Configure MCP Servers

Edit `config/mcp_config.json`:

```json
{
  "mcpServers": {
    "rosbag_reader": {
      "command": "/path_to_venv/bin/python",
      "args": [
        "/path_to_ws/mcp-rosbags/src/server.py"
      ],
    }
  }
}
```

### 5. Run the Application

```bash
python run.py
```

Open your browser to:
- **Main Lab**: http://localhost:8008
- **Analytics**: http://localhost:8008/analytics

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
- **Iterations**: Number of reasoning cycles (simple agent in a loop)

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
Add to `config/mcp_config.json`:
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

## Troubleshooting

### MCP Servers Not Connecting
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


### Debug Mode

Set `DEBUG=True` in `.env` for:
- Auto-reload on code changes
- Detailed logging
- Error stack traces

## License

Apache 2


## Acknowledgments

Built for testing MCP (Model Context Protocol) servers with various LLM providers. MCP is originally developed by Anthropic.
