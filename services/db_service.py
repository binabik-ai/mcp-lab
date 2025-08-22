"""
Database service for MCP Lab
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseService:
    """SQLite database service for conversation storage"""
    
    def __init__(self, db_path: str = "mcp_lab.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    ai_response TEXT,
                    
                    -- Metrics
                    total_tokens INTEGER,
                    total_latency REAL,
                    tool_iterations INTEGER,
                    tools_called_count INTEGER,
                    
                    -- Details (JSON)
                    tool_calls TEXT,
                    media_outputs TEXT
                )
            ''')
            
            # Create indices for better query performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_provider ON conversations(provider)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_model ON conversations(model)')
            
            conn.commit()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def save_conversation(self, session_id: str, provider: str, model: str,
                         user_message: str, ai_response: str,
                         total_tokens: int = 0, total_latency: float = 0,
                         tool_iterations: int = 0, tools_called_count: int = 0,
                         tool_calls: List[Dict] = None, media_outputs: List[str] = None):
        """Save a conversation to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO conversations 
                    (session_id, provider, model, user_message, ai_response,
                     total_tokens, total_latency, tool_iterations, tools_called_count,
                     tool_calls, media_outputs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id, provider, model, user_message, ai_response,
                    total_tokens, total_latency, tool_iterations, tools_called_count,
                    json.dumps(tool_calls) if tool_calls else None,
                    json.dumps(media_outputs) if media_outputs else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def get_conversations(self, limit: int = 100, offset: int = 0, 
                          filters: Dict[str, Any] = None) -> List[Dict]:
        """Get conversations with optional filtering"""
        query = '''
            SELECT id, session_id, timestamp, provider, model, 
                   user_message, ai_response, total_tokens, total_latency,
                   tool_iterations, tools_called_count, tool_calls
            FROM conversations
            WHERE 1=1
        '''
        params = []
        
        if filters:
            if 'provider' in filters:
                query += ' AND provider = ?'
                params.append(filters['provider'])
            if 'model' in filters:
                query += ' AND model = ?'
                params.append(filters['model'])
            if 'start_date' in filters:
                query += ' AND timestamp >= ?'
                params.append(filters['start_date'])
            if 'end_date' in filters:
                query += ' AND timestamp <= ?'
                params.append(filters['end_date'])
        
        query += ' ORDER BY timestamp DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                conversations = []
                for row in cursor:
                    conv = dict(row)
                    # Parse JSON fields
                    if conv['tool_calls']:
                        conv['tool_calls'] = json.loads(conv['tool_calls'])
                    conversations.append(conv)
                
                return conversations
        except Exception as e:
            logger.error(f"Error getting conversations: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analytics statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total conversations
                total = conn.execute('SELECT COUNT(*) FROM conversations').fetchone()[0]
                
                # Stats by provider
                provider_stats = {}
                cursor = conn.execute('''
                    SELECT provider, 
                           COUNT(*) as count,
                           AVG(total_latency) as avg_latency,
                           AVG(total_tokens) as avg_tokens,
                           AVG(tools_called_count) as avg_tools
                    FROM conversations
                    GROUP BY provider
                ''')
                for row in cursor:
                    provider_stats[row[0]] = {
                        'count': row[1],
                        'avg_latency': round(row[2], 2) if row[2] else 0,
                        'avg_tokens': round(row[3], 0) if row[3] else 0,
                        'avg_tools': round(row[4], 1) if row[4] else 0
                    }
                
                # Stats by model
                model_stats = {}
                cursor = conn.execute('''
                    SELECT model, 
                           COUNT(*) as count,
                           AVG(total_latency) as avg_latency,
                           AVG(total_tokens) as avg_tokens,
                           AVG(tools_called_count) as avg_tools
                    FROM conversations
                    GROUP BY model
                ''')
                for row in cursor:
                    model_stats[row[0]] = {
                        'count': row[1],
                        'avg_latency': round(row[2], 2) if row[2] else 0,
                        'avg_tokens': round(row[3], 0) if row[3] else 0,
                        'avg_tools': round(row[4], 1) if row[4] else 0
                    }
                
                return {
                    'total_conversations': total,
                    'by_provider': provider_stats,
                    'by_model': model_stats
                }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def clear_all(self):
        """Clear all conversations"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('DELETE FROM conversations')
                conn.commit()
            logger.info("Database cleared")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")