/**
 * KerasFactory Code Copy Functionality
 * Copy code snippets to clipboard with visual feedback
 */

class CodeCopy {
    constructor() {
        this.copyButtons = document.querySelectorAll('.copy-btn');
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.createToast();
    }
    
    bindEvents() {
        this.copyButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                this.copyCode(button);
            });
        });
        
        // Also handle copy buttons in dynamically loaded content
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('copy-btn')) {
                e.preventDefault();
                this.copyCode(e.target);
            }
        });
    }
    
    async copyCode(button) {
        try {
            const codeBlock = button.closest('.code-snippet').querySelector('pre code');
            const code = codeBlock.textContent;
            
            // Use modern clipboard API if available
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(code);
            } else {
                // Fallback for older browsers
                this.fallbackCopyTextToClipboard(code);
            }
            
            this.showCopyFeedback(button);
            
        } catch (err) {
            console.error('Failed to copy code: ', err);
            this.showCopyError(button);
        }
    }
    
    fallbackCopyTextToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
        } catch (err) {
            console.error('Fallback copy failed: ', err);
        }
        
        document.body.removeChild(textArea);
    }
    
    showCopyFeedback(button) {
        const originalText = button.innerHTML;
        const originalClass = button.className;
        
        // Update button appearance
        button.innerHTML = '✓ Copied!';
        button.className = originalClass + ' copied';
        
        // Show toast notification
        this.showToast('Code copied to clipboard!', 'success');
        
        // Reset button after 2 seconds
        setTimeout(() => {
            button.innerHTML = originalText;
            button.className = originalClass;
        }, 2000);
    }
    
    showCopyError(button) {
        const originalText = button.innerHTML;
        const originalClass = button.className;
        
        // Update button appearance
        button.innerHTML = '✗ Error';
        button.className = originalClass + ' error';
        
        // Show toast notification
        this.showToast('Failed to copy code. Please try again.', 'error');
        
        // Reset button after 2 seconds
        setTimeout(() => {
            button.innerHTML = originalText;
            button.className = originalClass;
        }, 2000);
    }
    
    createToast() {
        // Create toast container if it doesn't exist
        if (!document.getElementById('toast-container')) {
            const toastContainer = document.createElement('div');
            toastContainer.id = 'toast-container';
            toastContainer.className = 'toast-container';
            document.body.appendChild(toastContainer);
        }
    }
    
    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        toastContainer.appendChild(toast);
        
        // Animate in
        setTimeout(() => {
            toast.classList.add('show');
        }, 10);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.classList.add('hide');
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, 3000);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CodeCopy();
});

// Export for global access
window.CodeCopy = CodeCopy;
