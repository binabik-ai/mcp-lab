"""
Configuration for MCP Lab
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings"""
    
    # App info
    APP_NAME = "MCP Lab"
    VERSION = "1.0.0"
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Server settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '8000'))
    
    # API Keys (at least one required)
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
    
    # Default model settings
    DEFAULT_PROVIDER = os.getenv('DEFAULT_PROVIDER', 'groq')
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'llama-3.3-70b-versatile')
    MAX_TOKENS = int(os.getenv('MAX_TOKENS', '4096'))
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
    
    # MCP Configuration
    MCP_CONFIG_FILE = os.getenv('MCP_CONFIG_FILE', 'config/mcp_config.json')
    
    # Database
    DATABASE_PATH = os.getenv('DATABASE_PATH', 'mcp_lab.db')
    
    # UI Settings
    TRUNCATE_TOOL_OUTPUT = int(os.getenv('TRUNCATE_TOOL_OUTPUT', '500'))  # characters
    
    @classmethod
    def get_available_providers(cls):
        """Get list of configured providers"""
        providers = []
        if cls.GROQ_API_KEY:
            providers.append('groq')
        if cls.OPENAI_API_KEY:
            providers.append('openai')
        if cls.ANTHROPIC_API_KEY:
            providers.append('anthropic')
        return providers