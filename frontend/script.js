class MCPLab {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.currentProvider = 'groq';
        this.currentModel = null;
        this.debugMode = false;
        this.pendingModelChange = null;
        
        this.initElements();
        this.initWebSocket();
        this.initEventListeners();
        this.loadModels();
    }
    
    initElements() {
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.providerSelect = document.getElementById('providerSelect');
        this.modelSelect = document.getElementById('modelSelect');
        this.serverStatus = document.getElementById('serverStatus');
        this.debugCheckbox = document.getElementById('debugMode');
        this.modelChangeModal = document.getElementById('modelChangeModal');
        this.toolsInfo = document.getElementById('toolsInfo');
    }
    
    initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('Connected to MCP Lab');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            this.updateServerStatus('Disconnected', 'error');
            setTimeout(() => this.initWebSocket(), 3000);
        };
    }
    
    initEventListeners() {
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        this.providerSelect.addEventListener('change', (e) => {
            this.showModelChangeDialog(e.target.value, null);
        });
        
        this.modelSelect.addEventListener('change', (e) => {
            this.showModelChangeDialog(this.currentProvider, e.target.value);
        });
        
        this.debugCheckbox.addEventListener('change', (e) => {
            this.debugMode = e.target.checked;
        });
        
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 150) + 'px';
        });
    }
    
    async loadModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            
            const models = data.models[this.currentProvider] || [];
            this.modelSelect.innerHTML = models.map(model => 
                `<option value="${model}">${this.formatModelName(model)}</option>`
            ).join('');
            
            if (models.length > 0) {
                this.currentModel = models[0];
                this.modelSelect.value = this.currentModel;
            }
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }
    
    formatModelName(model) {
        // Prettier model names
        return model.replace(/-/g, ' ').replace(/_/g, ' ')
            .split(' ').map(word => 
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
    }
    
    handleMessage(data) {
        switch(data.type) {
            case 'connected':
                this.sessionId = data.session_id;
                if (data.mcp_status) {
                    this.updateMCPStatus(data.mcp_status);
                }
                break;
            
            case 'chat_start':
                this.addTypingIndicator();
                break;
            
            case 'chat_complete':
                this.removeTypingIndicator();
                this.addAIMessage(data.response, data.metrics, data.tool_calls, data.media_outputs);
                break;
            
            case 'error':
                this.removeTypingIndicator();
                this.showError(data.message);
                break;
            
            case 'model_changed':
                if (!data.context_kept) {
                    this.clearMessages();
                }
                break;
        }
    }
    
    updateMCPStatus(status) {
        const connected = status.connected_count || 0;
        const total = Object.keys(status.servers || {}).length;
        const tools = status.total_tools || 0;
        
        const statusText = `${connected}/${total} servers • ${tools} tools`;
        this.serverStatus.innerHTML = `<span class="status-dot ${connected > 0 ? 'connected' : 'error'}"></span> ${statusText}`;
        
        // Update welcome message
        if (this.toolsInfo) {
            this.toolsInfo.textContent = `Connected to ${connected} MCP servers with ${tools} available tools`;
        }
    }
    
    updateServerStatus(text, status = 'connected') {
        this.serverStatus.innerHTML = `<span class="status-dot ${status}"></span> ${text}`;
    }
    
    sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        
        // Hide welcome screen
        const welcome = document.querySelector('.welcome');
        if (welcome) welcome.style.display = 'none';
        
        // Add user message
        this.addUserMessage(message);
        
        // Clear input
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        
        // Send to server
        this.ws.send(JSON.stringify({
            type: 'chat_message',
            message: message,
            model: this.currentModel,
            provider: this.currentProvider,
            debug_mode: this.debugMode
        }));
    }
    
    addUserMessage(text) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message user';
        messageEl.innerHTML = `
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(text)}</div>
            </div>
        `;
        this.messagesContainer.appendChild(messageEl);
        this.scrollToBottom();
    }
    
    addAIMessage(text, metrics, toolCalls, mediaOutputs) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message ai';
        
        // Render markdown
        const renderedText = marked.parse(text);
        
        // Build tool calls display
        let toolCallsHtml = '';
        if (this.debugMode && toolCalls && toolCalls.length > 0) {
            toolCallsHtml = '<div class="tool-calls">';
            toolCallsHtml += '<div class="tool-calls-header">🔧 Tool Execution</div>';
            
            toolCalls.forEach((call, index) => {
                const args = JSON.stringify(call.args, null, 2);
                const result = call.full_result || call.result || call.error || 'No result';
                const resultId = `result-${this.sessionId}-${Date.now()}-${index}`;
                
                // Check if result is long
                const isLongResult = result.length > 500;
                
                toolCallsHtml += `
                    <div class="tool-call">
                        <div class="tool-name">${call.name} (${call.latency?.toFixed(2)}s)</div>
                        <details>
                            <summary>Arguments</summary>
                            <pre>${this.escapeHtml(args)}</pre>
                        </details>
                        <details open>
                            <summary>Result</summary>
                            <pre id="${resultId}" class="tool-result ${isLongResult ? 'collapsed' : ''}">${this.escapeHtml(result)}</pre>
                            ${isLongResult ? `<a class="tool-result-expand" onclick="toggleResult('${resultId}')">Show full result</a>` : ''}
                        </details>
                    </div>
                `;
            });
            
            toolCallsHtml += '</div>';
        }
        
        // Build media outputs from AI response
        let mediaHtml = '';
        if (mediaOutputs && mediaOutputs.length > 0) {
            mediaOutputs.forEach(media => {
                if (media.type === 'iframe') {
                    mediaHtml += `<iframe src="${media.url}" class="media-iframe"></iframe>`;
                } else if (media.type === 'image') {
                    mediaHtml += `<img src="${media.url}" class="media-image" />`;
                }
            });
        }
        
        // Build metrics display
        let metricsHtml = '';
        if (metrics) {
            metricsHtml = `
                <div class="metrics">
                    📊 ${metrics.tokens || 0} tokens • 
                    ⏱️ ${metrics.latency || 0}s • 
                    🔧 ${metrics.tools_called || 0} tools • 
                    🔄 ${metrics.iterations || 0} iteration${metrics.iterations !== 1 ? 's' : ''}
                </div>
            `;
        }
        
        messageEl.innerHTML = `
            <div class="message-content">
                ${toolCallsHtml}
                <div class="message-text">${renderedText}</div>
                ${mediaHtml}
                ${metricsHtml}
            </div>
        `;
        
        this.messagesContainer.appendChild(messageEl);
        this.scrollToBottom();
    }
    
    addTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'message ai typing-indicator';
        indicator.id = 'typing-indicator';
        indicator.innerHTML = `
            <div class="message-content">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        this.messagesContainer.appendChild(indicator);
        this.scrollToBottom();
    }
    
    removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
    }
    
    showError(message) {
        const errorEl = document.createElement('div');
        errorEl.className = 'message error';
        errorEl.innerHTML = `
            <div class="message-content">
                <div class="message-text">⚠️ Error: ${this.escapeHtml(message)}</div>
            </div>
        `;
        this.messagesContainer.appendChild(errorEl);
        this.scrollToBottom();
    }
    
    showModelChangeDialog(provider, model) {
        this.pendingModelChange = { provider, model };
        this.modelChangeModal.style.display = 'flex';
    }
    
    clearMessages() {
        const messages = this.messagesContainer.querySelectorAll('.message');
        messages.forEach(msg => msg.remove());
    }
    
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global functions for modal
let lab;

function toggleResult(resultId) {
    const resultEl = document.getElementById(resultId);
    const linkEl = resultEl.nextElementSibling;
    
    if (resultEl.classList.contains('collapsed')) {
        resultEl.classList.remove('collapsed');
        linkEl.textContent = 'Show less';
    } else {
        resultEl.classList.add('collapsed');
        linkEl.textContent = 'Show full result';
    }
}

function confirmModelChange(keepContext) {
    if (!lab.pendingModelChange) return;
    
    const { provider, model } = lab.pendingModelChange;
    
    if (provider && provider !== lab.currentProvider) {
        lab.currentProvider = provider;
        lab.providerSelect.value = provider;
        lab.loadModels().then(() => {
            if (lab.currentModel) {
                lab.ws.send(JSON.stringify({
                    type: 'change_model',
                    provider: provider,
                    model: lab.currentModel,
                    keep_context: keepContext
                }));
            }
        });
    } else if (model && model !== lab.currentModel) {
        lab.currentModel = model;
        lab.modelSelect.value = model;
        lab.ws.send(JSON.stringify({
            type: 'change_model',
            provider: lab.currentProvider,
            model: model,
            keep_context: keepContext
        }));
    }
    
    document.getElementById('modelChangeModal').style.display = 'none';
    lab.pendingModelChange = null;
}

function cancelModelChange() {
    // Reset selectors
    lab.providerSelect.value = lab.currentProvider;
    lab.modelSelect.value = lab.currentModel;
    document.getElementById('modelChangeModal').style.display = 'none';
    lab.pendingModelChange = null;
}

function clearChat() {
    if (confirm('Clear conversation history?')) {
        lab.clearMessages();
        lab.ws.send(JSON.stringify({ type: 'clear_history' }));
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    lab = new MCPLab();
});