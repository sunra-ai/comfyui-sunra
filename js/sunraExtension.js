/**
 * ComfyUI Sunra.ai Plugin Client Extension
 * 
 * Provides enhanced UI feedback and messaging for Sunra.ai nodes.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Extension configuration
const EXTENSION_NAME = "sunra.extension";
const MESSAGE_TYPES = {
    GENERATION_START: "sunra.generation.start",
    GENERATION_PROGRESS: "sunra.generation.progress", 
    GENERATION_COMPLETE: "sunra.generation.complete",
    GENERATION_ERROR: "sunra.generation.error",
    STATUS_UPDATE: "sunra.status.update"
};

// UI helpers
const createNotification = (message, type = "info", duration = 3000) => {
    const notification = document.createElement("div");
    notification.className = `sunra-notification sunra-notification-${type}`;
    notification.textContent = message;
    
    // Basic styling
    Object.assign(notification.style, {
        position: "fixed",
        top: "20px",
        right: "20px",
        padding: "12px 20px",
        borderRadius: "6px",
        color: "white",
        fontWeight: "500",
        zIndex: "10000",
        maxWidth: "400px",
        boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
        transition: "all 0.3s ease",
        backgroundColor: type === "error" ? "#e74c3c" : 
                        type === "success" ? "#27ae60" : 
                        type === "warning" ? "#f39c12" : "#3498db"
    });
    
    document.body.appendChild(notification);
    
    // Auto remove
    setTimeout(() => {
        notification.style.opacity = "0";
        notification.style.transform = "translateX(100%)";
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, duration);
};

const addProgressBar = (node, progress = 0) => {
    // Remove existing progress bar if any
    const existingBar = node.progressBar;
    if (existingBar && existingBar.parentNode) {
        existingBar.parentNode.removeChild(existingBar);
    }
    
    // Create new progress bar
    const progressBar = document.createElement("div");
    progressBar.className = "sunra-progress-bar";
    
    const progressFill = document.createElement("div");
    progressFill.className = "sunra-progress-fill";
    
    Object.assign(progressBar.style, {
        position: "absolute",
        bottom: "0",
        left: "0",
        right: "0",
        height: "4px",
        backgroundColor: "rgba(0,0,0,0.2)",
        borderRadius: "0 0 4px 4px",
        overflow: "hidden"
    });
    
    Object.assign(progressFill.style, {
        height: "100%",
        backgroundColor: "#3498db",
        width: `${progress * 100}%`,
        transition: "width 0.3s ease"
    });
    
    progressBar.appendChild(progressFill);
    
    // Attach to node element
    if (node.element) {
        node.element.style.position = "relative";
        node.element.appendChild(progressBar);
        node.progressBar = progressBar;
        node.progressFill = progressFill;
    }
};

const updateProgress = (node, progress) => {
    if (node.progressFill) {
        node.progressFill.style.width = `${Math.min(100, Math.max(0, progress * 100))}%`;
    }
};

const removeProgressBar = (node) => {
    if (node.progressBar && node.progressBar.parentNode) {
        node.progressBar.parentNode.removeChild(node.progressBar);
        delete node.progressBar;
        delete node.progressFill;
    }
};

// Register the extension
app.registerExtension({
    name: EXTENSION_NAME,
    
    async setup() {
        // Add CSS styles
        const style = document.createElement("style");
        style.textContent = `
            .sunra-notification {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                font-size: 14px;
                line-height: 1.4;
            }
            
            .sunra-progress-bar {
                pointer-events: none;
            }
            
            .sunra-node-status {
                position: absolute;
                top: 4px;
                right: 4px;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #95a5a6;
            }
            
            .sunra-node-status.processing {
                background-color: #f39c12;
                animation: sunra-pulse 1.5s infinite;
            }
            
            .sunra-node-status.completed {
                background-color: #27ae60;
            }
            
            .sunra-node-status.error {
                background-color: #e74c3c;
            }
            
            @keyframes sunra-pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
        `;
        document.head.appendChild(style);
        
        // Register message handlers
        api.addEventListener(MESSAGE_TYPES.GENERATION_START, (event) => {
            const { node_id, model } = event.detail;
            createNotification(`Starting generation with ${model}...`, "info");
            
            const node = app.graph.getNodeById(node_id);
            if (node) {
                addProgressBar(node, 0);
            }
        });
        
        api.addEventListener(MESSAGE_TYPES.GENERATION_PROGRESS, (event) => {
            const { node_id, progress, step, total_steps } = event.detail;
            const node = app.graph.getNodeById(node_id);
            
            if (node) {
                updateProgress(node, progress);
                
                // Update node title with progress
                if (step && total_steps) {
                    node.title = `${node.constructor.title || node.type} (${step}/${total_steps})`;
                }
            }
        });
        
        api.addEventListener(MESSAGE_TYPES.GENERATION_COMPLETE, (event) => {
            const { node_id, message } = event.detail;
            createNotification(message || "Generation completed!", "success");
            
            const node = app.graph.getNodeById(node_id);
            if (node) {
                removeProgressBar(node);
                // Restore original title
                node.title = node.constructor.title || node.type;
            }
        });
        
        api.addEventListener(MESSAGE_TYPES.GENERATION_ERROR, (event) => {
            const { node_id, error } = event.detail;
            createNotification(`Error: ${error}`, "error", 5000);
            
            const node = app.graph.getNodeById(node_id);
            if (node) {
                removeProgressBar(node);
                // Restore original title
                node.title = node.constructor.title || node.type;
            }
        });
        
        api.addEventListener(MESSAGE_TYPES.STATUS_UPDATE, (event) => {
            const { node_id, status, message } = event.detail;
            
            const node = app.graph.getNodeById(node_id);
            if (node && status) {
                // Add status indicator
                let statusIndicator = node.element?.querySelector('.sunra-node-status');
                if (!statusIndicator) {
                    statusIndicator = document.createElement('div');
                    statusIndicator.className = 'sunra-node-status';
                    if (node.element) {
                        node.element.style.position = 'relative';
                        node.element.appendChild(statusIndicator);
                    }
                }
                
                // Update status class
                statusIndicator.className = `sunra-node-status ${status}`;
                
                if (message) {
                    createNotification(message, status === "error" ? "error" : "info");
                }
            }
        });
    },
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Enhance Sunra.ai nodes
        if (nodeData.name.startsWith("Sunra")) {
            // Add custom properties or behaviors
            const originalExecute = nodeType.prototype.onExecute;
            if (originalExecute) {
                nodeType.prototype.onExecute = function(...args) {
                    // Send start message
                    api.dispatchEvent(new CustomEvent(MESSAGE_TYPES.GENERATION_START, {
                        detail: {
                            node_id: this.id,
                            model: this.properties?.model || "unknown"
                        }
                    }));
                    
                    return originalExecute.apply(this, args);
                };
            }
        }
    },
    
    async nodeCreated(node) {
        // Add any node-specific initialization for Sunra nodes
        if (node.type.startsWith("Sunra")) {
            // Add helpful tooltips or UI enhancements
            if (node.widgets) {
                node.widgets.forEach(widget => {
                    if (widget.name === "seed" && widget.value === -1) {
                        widget.tooltip = "Use -1 for random seed";
                    }
                    if (widget.name === "model") {
                        widget.tooltip = "Choose the model variant based on your needs:\n" +
                                        "• dev: Fast, experimental features\n" + 
                                        "• pro: Balanced speed and quality\n" +
                                        "• max: Highest quality, slower";
                    }
                });
            }
        }
    }
});

// Export for potential use by other extensions
window.SunraExtension = {
    createNotification,
    addProgressBar,
    updateProgress,
    removeProgressBar,
    MESSAGE_TYPES
}; 