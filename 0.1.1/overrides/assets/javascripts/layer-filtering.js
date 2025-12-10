/**
 * KerasFactory Layer Filtering System
 * Interactive search and filtering for layer documentation
 */

class LayerFilter {
    constructor() {
        this.searchInput = document.getElementById('layer-search');
        this.filterButtons = document.querySelectorAll('.filter-button');
        this.viewButtons = document.querySelectorAll('.view-btn');
        this.layerCards = document.querySelectorAll('.layer-card');
        this.resultCount = document.getElementById('result-count');
        
        this.currentFilters = {
            search: '',
            category: 'all',
            complexity: 'all',
            useCase: 'all',
            performance: 'all'
        };
        
        this.currentView = 'grid';
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.updateURL();
        this.applyFilters();
    }
    
    bindEvents() {
        // Search input
        if (this.searchInput) {
            this.searchInput.addEventListener('input', (e) => {
                this.currentFilters.search = e.target.value.toLowerCase();
                this.applyFilters();
                this.updateURL();
            });
        }
        
        // Filter buttons
        this.filterButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const filterType = e.target.dataset.category || 
                                 e.target.dataset.complexity || 
                                 e.target.dataset.useCase ||
                                 e.target.dataset.performance;
                const filterValue = e.target.textContent.trim();
                
                if (e.target.dataset.category) {
                    this.currentFilters.category = filterType;
                } else if (e.target.dataset.complexity) {
                    this.currentFilters.complexity = filterType;
                } else if (e.target.dataset.useCase) {
                    this.currentFilters.useCase = filterType;
                } else if (e.target.dataset.performance) {
                    this.currentFilters.performance = filterType;
                }
                
                this.updateFilterButtons(e.target);
                this.applyFilters();
                this.updateURL();
            });
        });
        
        // View toggle buttons
        this.viewButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                this.currentView = e.target.dataset.view;
                this.updateViewButtons(e.target);
                this.applyView();
                this.updateURL();
            });
        });
        
        // Clear search button
        const clearButton = document.querySelector('.search-clear');
        if (clearButton) {
            clearButton.addEventListener('click', () => {
                this.clearSearch();
            });
        }
        
        // Load filters from URL on page load
        this.loadFromURL();
    }
    
    updateFilterButtons(activeButton) {
        // Remove active class from all buttons in the same group
        const group = activeButton.closest('.filter-group');
        group.querySelectorAll('.filter-button').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Add active class to clicked button
        activeButton.classList.add('active');
    }
    
    updateViewButtons(activeButton) {
        this.viewButtons.forEach(btn => {
            btn.classList.remove('active');
        });
        activeButton.classList.add('active');
    }
    
    applyFilters() {
        let visibleCount = 0;
        
        this.layerCards.forEach(card => {
            const isVisible = this.isCardVisible(card);
            card.style.display = isVisible ? 'block' : 'none';
            
            if (isVisible) {
                visibleCount++;
            }
        });
        
        this.updateResultCount(visibleCount);
    }
    
    isCardVisible(card) {
        const cardData = {
            category: card.dataset.category || '',
            complexity: card.dataset.complexity || '',
            useCase: card.dataset.useCase || '',
            performance: card.dataset.performance || ''
        };
        
        const text = card.textContent.toLowerCase();
        
        // Search filter
        if (this.currentFilters.search && !text.includes(this.currentFilters.search)) {
            return false;
        }
        
        // Category filter
        if (this.currentFilters.category !== 'all' && 
            cardData.category !== this.currentFilters.category) {
            return false;
        }
        
        // Complexity filter
        if (this.currentFilters.complexity !== 'all' && 
            cardData.complexity !== this.currentFilters.complexity) {
            return false;
        }
        
        // Use case filter
        if (this.currentFilters.useCase !== 'all' && 
            cardData.useCase !== this.currentFilters.useCase) {
            return false;
        }
        
        // Performance filter
        if (this.currentFilters.performance !== 'all' && 
            cardData.performance !== this.currentFilters.performance) {
            return false;
        }
        
        return true;
    }
    
    applyView() {
        const container = document.querySelector('.layers-container') || 
                        document.querySelector('.md-grid') || 
                        document.querySelector('.popular-layers');
        
        if (!container) return;
        
        if (this.currentView === 'list') {
            container.classList.add('list-view');
            container.classList.remove('grid-view');
        } else {
            container.classList.add('grid-view');
            container.classList.remove('list-view');
        }
    }
    
    updateResultCount(count) {
        if (this.resultCount) {
            const total = this.layerCards.length;
            this.resultCount.textContent = `Showing ${count} of ${total} layers`;
        }
    }
    
    clearSearch() {
        if (this.searchInput) {
            this.searchInput.value = '';
            this.currentFilters.search = '';
            this.applyFilters();
            this.updateURL();
        }
    }
    
    updateURL() {
        const params = new URLSearchParams();
        
        if (this.currentFilters.search) {
            params.set('search', this.currentFilters.search);
        }
        if (this.currentFilters.category !== 'all') {
            params.set('category', this.currentFilters.category);
        }
        if (this.currentFilters.complexity !== 'all') {
            params.set('complexity', this.currentFilters.complexity);
        }
        if (this.currentFilters.useCase !== 'all') {
            params.set('useCase', this.currentFilters.useCase);
        }
        if (this.currentFilters.performance !== 'all') {
            params.set('performance', this.currentFilters.performance);
        }
        if (this.currentView !== 'grid') {
            params.set('view', this.currentView);
        }
        
        const newURL = params.toString() ? 
            `${window.location.pathname}?${params.toString()}` : 
            window.location.pathname;
        
        window.history.replaceState({}, '', newURL);
    }
    
    loadFromURL() {
        const params = new URLSearchParams(window.location.search);
        
        // Load search
        const search = params.get('search');
        if (search && this.searchInput) {
            this.searchInput.value = search;
            this.currentFilters.search = search.toLowerCase();
        }
        
        // Load filters
        const category = params.get('category');
        if (category) {
            this.currentFilters.category = category;
            this.updateFilterButton('category', category);
        }
        
        const complexity = params.get('complexity');
        if (complexity) {
            this.currentFilters.complexity = complexity;
            this.updateFilterButton('complexity', complexity);
        }
        
        const useCase = params.get('useCase');
        if (useCase) {
            this.currentFilters.useCase = useCase;
            this.updateFilterButton('useCase', useCase);
        }
        
        const performance = params.get('performance');
        if (performance) {
            this.currentFilters.performance = performance;
            this.updateFilterButton('performance', performance);
        }
        
        // Load view
        const view = params.get('view');
        if (view) {
            this.currentView = view;
            this.updateViewButton(view);
        }
    }
    
    updateFilterButton(type, value) {
        const button = document.querySelector(`[data-${type}="${value}"]`);
        if (button) {
            this.updateFilterButtons(button);
        }
    }
    
    updateViewButton(view) {
        const button = document.querySelector(`[data-view="${view}"]`);
        if (button) {
            this.updateViewButtons(button);
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new LayerFilter();
});

// Export for global access
window.LayerFilter = LayerFilter;
