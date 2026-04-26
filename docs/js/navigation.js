// QuASIM Technical Portal - Navigation & Keyboard Shortcuts

class KeyboardShortcuts {
  constructor(shortcuts) {
    this.shortcuts = shortcuts;
    this.init();
  }
  
  init() {
    document.addEventListener('keydown', (e) => this.handleKeydown(e));
  }
  
  handleKeydown(e) {
    // Check if user is typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
      return;
    }
    
    const key = e.key.toLowerCase();
    const ctrl = e.ctrlKey || e.metaKey;
    const shift = e.shiftKey;
    
    this.shortcuts.forEach(shortcut => {
      if (shortcut.key === key && 
          (shortcut.ctrl === undefined || shortcut.ctrl === ctrl) &&
          (shortcut.shift === undefined || shortcut.shift === shift)) {
        e.preventDefault();
        shortcut.action();
      }
    });
  }
}

class TableOfContents {
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      selector: options.selector || 'h2, h3',
      title: options.title || 'Table of Contents'
    };
    
    this.generate();
  }
  
  generate() {
    const headings = document.querySelectorAll(this.options.selector);
    if (headings.length === 0) return;
    
    const nav = document.createElement('nav');
    nav.className = 'table-of-contents';
    nav.setAttribute('aria-label', 'Table of Contents');
    
    const title = document.createElement('h3');
    title.textContent = this.options.title;
    nav.appendChild(title);
    
    const list = document.createElement('ul');
    
    headings.forEach((heading, index) => {
      if (!heading.id) {
        heading.id = `heading-${index}`;
      }
      
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = `#${heading.id}`;
      link.textContent = heading.textContent;
      link.addEventListener('click', (e) => {
        e.preventDefault();
        heading.scrollIntoView({ behavior: 'smooth' });
      });
      
      item.appendChild(link);
      list.appendChild(item);
    });
    
    nav.appendChild(list);
    this.container.appendChild(nav);
  }
}

class ReadingProgress {
  constructor() {
    this.bar = document.createElement('div');
    this.bar.className = 'reading-progress';
    this.bar.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--quantum-green), var(--plasma-cyan));
      z-index: 1000;
      transition: width 0.1s ease;
      width: 0%;
    `;
    document.body.appendChild(this.bar);
    
    window.addEventListener('scroll', () => this.update());
  }
  
  update() {
    const windowHeight = window.innerHeight;
    const documentHeight = document.documentElement.scrollHeight;
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const progress = (scrollTop / (documentHeight - windowHeight)) * 100;
    this.bar.style.width = Math.min(progress, 100) + '%';
  }
}

class BackToTop {
  constructor() {
    this.button = document.createElement('button');
    this.button.className = 'back-to-top';
    this.button.innerHTML = '↑';
    this.button.setAttribute('aria-label', 'Back to top');
    this.button.style.cssText = `
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      width: 3rem;
      height: 3rem;
      border-radius: 50%;
      background-color: var(--quantum-green);
      color: var(--void-black);
      border: none;
      font-size: 1.5rem;
      cursor: pointer;
      opacity: 0;
      visibility: hidden;
      transition: all 0.3s ease;
      z-index: 100;
    `;
    
    document.body.appendChild(this.button);
    
    window.addEventListener('scroll', () => {
      if (window.pageYOffset > 300) {
        this.button.style.opacity = '1';
        this.button.style.visibility = 'visible';
      } else {
        this.button.style.opacity = '0';
        this.button.style.visibility = 'hidden';
      }
    });
    
    this.button.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  // Keyboard shortcuts
  new KeyboardShortcuts([
    { key: '?', action: () => alert('Keyboard Shortcuts:\n? - Show help\n/ - Focus search\ng+h - Go home\ng+a - Go to architecture\ng+p - Go to performance\ng+c - Go to compliance') },
    { key: '/', action: () => { const search = document.getElementById('search'); if (search) search.focus(); }},
    { key: 'g', shift: false, action: () => {} } // Placeholder for g+ shortcuts
  ]);
  
  // Reading progress
  new ReadingProgress();
  
  // Back to top
  new BackToTop();
});
