"""
MCP Server integration service with improved stdio communication
"""
import json
import asyncio
import subprocess
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class MCPServerState(Enum):
    """MCP Server connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"

class MCPServer:
    """Represents a single MCP server process with enhanced communication"""
    
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.process = None
        self._request_id = 0
        self._pending_requests = {}
        self._reader_task = None
        self.state = MCPServerState.DISCONNECTED
        self.last_error = None
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
        # Communication settings
        self.request_timeout = 30.0  # Default timeout for requests
        self.init_timeout = 10.0     # Timeout for initialization
        
    async def start(self):
        """Start the MCP server process with retry logic"""
        self.connection_attempts = 0
        
        while self.connection_attempts < self.max_connection_attempts:
            try:
                self.connection_attempts += 1
                self.state = MCPServerState.CONNECTING
                logger.info(f"Starting MCP server: {self.name} (attempt {self.connection_attempts}/{self.max_connection_attempts})")
                
                # Build command - handle both single command and command+args format
                if 'command' in self.config and 'args' in self.config:
                    # New format with separate command and args
                    cmd = [self.config['command']] + self.config.get('args', [])
                else:
                    # Old format with command as list
                    cmd = self.config.get('command', [])
                
                if not cmd:
                    raise ValueError(f"No command specified for MCP server {self.name}")
                
                # Merge environment variables
                env = {**os.environ.copy(), **self.config.get('env', {})}
                
                # Start process
                self.process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                
                logger.info(f"Started MCP server: {self.name} with PID {self.process.pid}")
                
                # Start response reader
                self._reader_task = asyncio.create_task(self._read_responses())
                
                # Start stderr reader for debugging
                asyncio.create_task(self._read_stderr())
                
                # Give server time to start
                await asyncio.sleep(0.5)
                
                # Initialize connection
                await self._initialize_connection()
                
                self.state = MCPServerState.CONNECTED
                logger.info(f"Successfully connected to MCP server: {self.name}")
                return
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout connecting to {self.name} (attempt {self.connection_attempts})")
                self.last_error = "Connection timeout"
                await self._cleanup()
                
            except Exception as e:
                logger.error(f"Failed to start MCP server {self.name} (attempt {self.connection_attempts}): {e}")
                self.last_error = str(e)
                await self._cleanup()
            
            if self.connection_attempts < self.max_connection_attempts:
                await asyncio.sleep(2)  # Wait before retry
        
        self.state = MCPServerState.ERROR
        raise RuntimeError(f"Failed to start MCP server {self.name} after {self.max_connection_attempts} attempts")
    
    async def _cleanup(self):
        """Clean up resources"""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            except ProcessLookupError:
                pass
            self.process = None
    
    async def _initialize_connection(self):
        """Initialize JSON-RPC connection over stdio"""
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {},
                "clientInfo": {
                    "name": "binabik-mcp-client",
                    "version": "1.0.0"
                }
            },
            "id": self._get_next_id()
        }
        
        try:
            response = await asyncio.wait_for(
                self._send_request(init_request), 
                timeout=self.init_timeout
            )
            logger.info(f"MCP server {self.name} initialized successfully")
            logger.debug(f"Init response: {response}")
            
            # Send initialized notification
            await self._send_notification({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            })
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Timeout initializing {self.name}")
        except Exception as e:
            raise RuntimeError(f"Error initializing {self.name}: {e}")
    
    def _get_next_id(self) -> int:
        """Get next request ID"""
        self._request_id += 1
        return self._request_id
    
    async def _send_request(self, request: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """Send a request and wait for response with timeout"""
        if self.state != MCPServerState.CONNECTED and request.get("method") != "initialize":
            raise RuntimeError(f"MCP server '{self.name}' is not connected (state: {self.state})")
        
        request_id = request.get("id")
        
        # Create future for this request
        future = asyncio.Future()
        if request_id is not None:
            self._pending_requests[request_id] = future
        
        try:
            # Send the request
            await self._write_message(request)
            
            # Wait for response if this is a request (not a notification)
            if request_id is not None:
                timeout = timeout or self.request_timeout
                return await asyncio.wait_for(future, timeout=timeout)
                
        except asyncio.TimeoutError:
            # Clean up pending request
            if request_id in self._pending_requests:
                del self._pending_requests[request_id]
            raise
        except Exception:
            # Clean up on any error
            if request_id in self._pending_requests:
                del self._pending_requests[request_id]
            raise
    
    async def _send_notification(self, notification: Dict[str, Any]):
        """Send a notification (no response expected)"""
        await self._write_message(notification)
    
    async def _write_message(self, message: Dict[str, Any]):
        """Write a message to the server's stdin"""
        if not self.process or self.process.returncode is not None:
            raise RuntimeError(f"MCP server '{self.name}' is not running")
        
        message_str = json.dumps(message) + "\n"
        logger.debug(f"[{self.name} →] {message_str.strip()[:200]}")
        
        try:
            self.process.stdin.write(message_str.encode())
            await self.process.stdin.drain()
        except Exception as e:
            logger.error(f"Failed to write to {self.name}: {e}")
            self.state = MCPServerState.ERROR
            raise
    
    async def _read_responses(self):
        """Continuously read responses from the server"""
        logger.debug(f"Starting response reader for {self.name}")
        
        while self.process and self.process.returncode is None:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    logger.info(f"EOF from {self.name} stdout")
                    break
                
                line_str = line.decode().strip()
                if not line_str:
                    continue
                
                logger.debug(f"[{self.name} ←] {line_str[:500]}")
                
                # Parse JSON response
                try:
                    response = json.loads(line_str)
                    
                    # Handle response with ID
                    if "id" in response:
                        request_id = response["id"]
                        if request_id in self._pending_requests:
                            future = self._pending_requests.pop(request_id)
                            if not future.done():
                                if "error" in response:
                                    error_info = response["error"]
                                    error_msg = f"MCP Error: {error_info.get('message', 'Unknown error')}"
                                    if 'data' in error_info:
                                        error_msg += f" - {error_info['data']}"
                                    future.set_exception(RuntimeError(error_msg))
                                else:
                                    future.set_result(response.get("result", {}))
                        else:
                            logger.warning(f"Received response for unknown request ID: {request_id}")
                    
                    # Handle notifications from server
                    elif "method" in response:
                        logger.debug(f"Notification from {self.name}: {response.get('method')}")
                        # Could handle specific notifications here if needed
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Non-JSON output from {self.name}: {line_str[:100]}")
                    
            except asyncio.CancelledError:
                logger.debug(f"Response reader cancelled for {self.name}")
                break
            except Exception as e:
                logger.error(f"Error reading from {self.name}: {e}")
                self.state = MCPServerState.ERROR
                break
        
        # Mark server as disconnected if we exit the loop
        if self.state == MCPServerState.CONNECTED:
            self.state = MCPServerState.DISCONNECTED
            logger.warning(f"MCP server {self.name} disconnected")
    
    async def _read_stderr(self):
        """Read stderr for debugging"""
        while self.process and self.process.returncode is None:
            try:
                line = await self.process.stderr.readline()
                if not line:
                    break
                
                error_str = line.decode().strip()
                if error_str:
                    logger.info(f"[{self.name} stderr] {error_str}")
                    
            except Exception as e:
                logger.debug(f"Error reading stderr from {self.name}: {e}")
                break
    
    async def call_tool(self, tool: str, params: dict, timeout: Optional[float] = None) -> Any:
        """Call a tool on this server with timeout"""
        if self.state != MCPServerState.CONNECTED:
            raise RuntimeError(f"Cannot call tool on disconnected server '{self.name}' (state: {self.state})")
        
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool,
                "arguments": params
            },
            "id": self._get_next_id()
        }
        
        try:
            response = await self._send_request(request, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling tool {tool} on server {self.name}")
            raise
        except Exception as e:
            logger.error(f"Error calling tool {tool} on server {self.name}: {e}")
            raise
    
    async def get_tools(self) -> List[dict]:
        """Get available tools from this server"""
        if self.state != MCPServerState.CONNECTED:
            logger.warning(f"Cannot get tools from disconnected server '{self.name}'")
            return []
        
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self._get_next_id()
        }
        
        try:
            response = await self._send_request(request, timeout=5.0)
            return response.get("tools", [])
        except Exception as e:
            logger.error(f"Error getting tools from {self.name}: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if the server is healthy"""
        if self.state != MCPServerState.CONNECTED:
            return False
        
        try:
            # Send a simple request to check connectivity
            tools = await self.get_tools()
            return True
        except:
            return False
    
    async def shutdown(self):
        """Shutdown the MCP server process gracefully"""
        logger.info(f"Shutting down MCP server: {self.name}")
        self.state = MCPServerState.SHUTTING_DOWN
        
        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()
        
        await self._cleanup()
        self.state = MCPServerState.DISCONNECTED

class MCPService:
    """Service for managing multiple MCP servers with enhanced monitoring"""
    
    def __init__(self, config):
        self.config = config
        self.servers: Dict[str, MCPServer] = {}
        self.config_file = Path(config.MCP_CONFIG_FILE)
        self._health_check_task = None
        self.health_check_interval = 30.0  # seconds
        
    async def initialize_servers(self):
        """Initialize all configured MCP servers"""
        if not self.config_file.exists():
            logger.warning(f"MCP configuration file not found: {self.config_file}")
            return
        
        try:
            with open(self.config_file) as f:
                mcp_config = json.load(f)
            
            logger.info(f"Loading MCP config with {len(mcp_config.get('mcpServers', {}))} servers")
            
            # Start servers in parallel with error handling
            tasks = []
            for name, server_config in mcp_config.get("mcpServers", {}).items():
                logger.info(f"Preparing to start MCP server: {name}")
                server = MCPServer(name, server_config)
                self.servers[name] = server
                tasks.append(self._start_server_with_error_handling(server))
            
            # Wait for all servers to start
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            successful = 0
            for name, result in zip(self.servers.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to start server {name}: {result}")
                    # Keep the server in the dict but marked as errored
                else:
                    successful += 1
                    # List tools for successfully started servers
                    server = self.servers[name]
                    if server.state == MCPServerState.CONNECTED:
                        tools = await server.get_tools()
                        logger.info(f"Server '{name}' provides {len(tools)} tools:")
                        for tool in tools:
                            logger.info(f"  - {tool.get('name')}: {tool.get('description', 'No description')[:60]}...")
            
            logger.info(f"Successfully started {successful}/{len(self.servers)} MCP servers")
            
            # Start health monitoring
            if successful > 0:
                self._health_check_task = asyncio.create_task(self._monitor_health())
                
        except Exception as e:
            logger.error(f"Failed to initialize MCP servers: {e}", exc_info=True)
    
    async def _start_server_with_error_handling(self, server: MCPServer):
        """Start a server with error handling"""
        try:
            await server.start()
        except Exception as e:
            logger.error(f"Error starting server {server.name}: {e}")
            return e
    
    async def _monitor_health(self):
        """Monitor health of all servers"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                for name, server in self.servers.items():
                    if server.state == MCPServerState.CONNECTED:
                        healthy = await server.health_check()
                        if not healthy:
                            logger.warning(f"Server {name} health check failed")
                            # Could implement auto-restart here if desired
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
    
    async def call_tool(self, server: str, tool: str, params: Dict[str, Any]) -> Any:
        """Call a tool on a specific MCP server"""
        logger.info(f"MCP TOOL CALL: {server}.{tool}")
        logger.debug(f"  Parameters: {json.dumps(params, indent=2)}")
        
        if server not in self.servers:
            raise ValueError(f"MCP server '{server}' not found")
        
        server_instance = self.servers[server]
        if server_instance.state != MCPServerState.CONNECTED:
            raise RuntimeError(f"MCP server '{server}' is not connected (state: {server_instance.state})")
        
        try:
            result = await server_instance.call_tool(tool, params)
            logger.debug(f"  Result: {json.dumps(result, indent=2) if isinstance(result, dict) else str(result)[:500]}")
            return result
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            raise
    
    async def get_available_tools(self, server: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get available tools from MCP servers"""
        tools = []
        
        if server:
            # Get tools from specific server
            if server in self.servers:
                server_instance = self.servers[server]
                if server_instance.state == MCPServerState.CONNECTED:
                    server_tools = await server_instance.get_tools()
                    for tool in server_tools:
                        tool['server'] = server
                        tools.append(tool)
        else:
            # Get tools from all connected servers
            for server_name, server_instance in self.servers.items():
                if server_instance.state == MCPServerState.CONNECTED:
                    try:
                        server_tools = await server_instance.get_tools()
                        for tool in server_tools:
                            tool['server'] = server_name
                            tools.append(tool)
                    except Exception as e:
                        logger.warning(f"Failed to get tools from {server_name}: {e}")
        
        return tools
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get status of all MCP servers"""
        status = {}
        for name, server in self.servers.items():
            status[name] = {
                "state": server.state.value,
                "running": server.process is not None and server.process.returncode is None,
                "pid": server.process.pid if server.process else None,
                "last_error": server.last_error,
                "connection_attempts": server.connection_attempts
            }
        return status
    
    async def restart_server(self, server_name: str) -> bool:
        """Restart a specific server"""
        if server_name not in self.servers:
            return False
        
        server = self.servers[server_name]
        
        # Shutdown existing
        await server.shutdown()
        
        # Start again
        try:
            await server.start()
            return True
        except Exception as e:
            logger.error(f"Failed to restart server {server_name}: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown all MCP servers"""
        logger.info("Shutting down all MCP servers...")
        
        # Cancel health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown all servers in parallel
        tasks = [server.shutdown() for server in self.servers.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.servers.clear()
        logger.info("All MCP servers shut down")