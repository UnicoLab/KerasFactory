/**
 * KerasFactory Tabs Functionality
 * Handle tab navigation for examples and content sections
 */

class TabManager {
    constructor() {
        this.tabContainers = document.querySelectorAll('.example-tabs, .tab-container');
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.initializeTabs();
    }
    
    bindEvents() {
        // Handle tab button clicks
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('tab-btn')) {
                e.preventDefault();
                this.switchTab(e.target);
            }
        });
        
        // Handle keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.target.classList.contains('tab-btn')) {
                this.handleKeyboardNavigation(e);
            }
        });
    }
    
    initializeTabs() {
        this.tabContainers.forEach(container => {
            const tabs = container.querySelectorAll('.tab-btn');
            const contents = container.querySelectorAll('.tab-content');
            
            // Set up ARIA attributes
            tabs.forEach((tab, index) => {
                tab.setAttribute('role', 'tab');
                tab.setAttribute('tabindex', index === 0 ? '0' : '-1');
                tab.setAttribute('aria-selected', index === 0 ? 'true' : 'false');
                
                const targetId = tab.getAttribute('data-tab');
                if (targetId) {
                    tab.setAttribute('aria-controls', targetId);
                }
            });
            
            contents.forEach((content, index) => {
                content.setAttribute('role', 'tabpanel');
                content.setAttribute('tabindex', '0');
                
                if (index === 0) {
                    content.classList.add('active');
                } else {
                    content.classList.remove('active');
                }
            });
            
            // Set up tab list
            const tabList = container.querySelector('.tab-buttons, .example-tabs');
            if (tabList) {
                tabList.setAttribute('role', 'tablist');
            }
        });
    }
    
    switchTab(clickedTab) {
        const container = clickedTab.closest('.example-tabs, .tab-container');
        if (!container) return;
        
        const targetTab = clickedTab.getAttribute('data-tab');
        if (!targetTab) return;
        
        // Update tab buttons
        const allTabs = container.querySelectorAll('.tab-btn');
        allTabs.forEach(tab => {
            tab.classList.remove('active');
            tab.setAttribute('aria-selected', 'false');
            tab.setAttribute('tabindex', '-1');
        });
        
        clickedTab.classList.add('active');
        clickedTab.setAttribute('aria-selected', 'true');
        clickedTab.setAttribute('tabindex', '0');
        
        // Update tab content
        const allContents = container.querySelectorAll('.tab-content');
        allContents.forEach(content => {
            content.classList.remove('active');
        });
        
        const targetContent = container.querySelector(`#${targetTab}`);
        if (targetContent) {
            targetContent.classList.add('active');
        }
        
        // Focus the content area for accessibility
        if (targetContent) {
            targetContent.focus();
        }
    }
    
    handleKeyboardNavigation(e) {
        const currentTab = e.target;
        const container = currentTab.closest('.example-tabs, .tab-container');
        if (!container) return;
        
        const tabs = Array.from(container.querySelectorAll('.tab-btn'));
        const currentIndex = tabs.indexOf(currentTab);
        
        let newIndex = currentIndex;
        
        switch (e.key) {
            case 'ArrowLeft':
            case 'ArrowUp':
                e.preventDefault();
                newIndex = currentIndex > 0 ? currentIndex - 1 : tabs.length - 1;
                break;
            case 'ArrowRight':
            case 'ArrowDown':
                e.preventDefault();
                newIndex = currentIndex < tabs.length - 1 ? currentIndex + 1 : 0;
                break;
            case 'Home':
                e.preventDefault();
                newIndex = 0;
                break;
            case 'End':
                e.preventDefault();
                newIndex = tabs.length - 1;
                break;
            case 'Enter':
            case ' ':
                e.preventDefault();
                this.switchTab(currentTab);
                return;
        }
        
        if (newIndex !== currentIndex) {
            tabs[newIndex].focus();
        }
    }
    
    // Method to programmatically switch to a specific tab
    switchToTab(container, tabId) {
        const tab = container.querySelector(`[data-tab="${tabId}"]`);
        if (tab) {
            this.switchTab(tab);
        }
    }
    
    // Method to get current active tab
    getActiveTab(container) {
        const activeTab = container.querySelector('.tab-btn.active');
        return activeTab ? activeTab.getAttribute('data-tab') : null;
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TabManager();
});

// Export for global access
window.TabManager = TabManager;
