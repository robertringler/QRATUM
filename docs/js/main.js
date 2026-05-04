// QuASIM Technical Portal - Main JavaScript
// Core initialization, navigation, scroll animations, counters, tabs, accordions, etc.

(function() {
  'use strict';

  // ============================================================================
  // Core Initialization
  // ============================================================================

  document.addEventListener('DOMContentLoaded', function() {
    initNavigation();
    initScrollAnimations();
    initCounters();
    initTabs();
    initAccordions();
    initCodeBlocks();
    initTooltips();
    initModals();
    initDropdowns();
    initBackToTop();
    initReadingProgress();
    initMobileNav();
  });

  // ============================================================================
  // Navigation
  // ============================================================================

  function initNavigation() {
    const navLinks = document.querySelectorAll('.navbar-nav a');
    const currentPath = window.location.pathname.split('/').pop() || 'index.html';

    navLinks.forEach(link => {
      const href = link.getAttribute('href');
      if (href === currentPath || (href === 'index.html' && currentPath === '')) {
        link.classList.add('active');
      }

      link.addEventListener('click', function(e) {
        if (this.getAttribute('href').startsWith('#')) {
          e.preventDefault();
          const target = document.querySelector(this.getAttribute('href'));
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }
      });
    });
  }

  // ============================================================================
  // Mobile Navigation
  // ============================================================================

  function initMobileNav() {
    const toggle = document.querySelector('.navbar-toggle');
    const nav = document.querySelector('.navbar-nav');

    if (toggle && nav) {
      toggle.addEventListener('click', function() {
        nav.classList.toggle('open');
        this.setAttribute('aria-expanded', nav.classList.contains('open'));
      });

      // Close on click outside
      document.addEventListener('click', function(e) {
        if (!toggle.contains(e.target) && !nav.contains(e.target)) {
          nav.classList.remove('open');
          toggle.setAttribute('aria-expanded', 'false');
        }
      });

      // Close on escape
      document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && nav.classList.contains('open')) {
          nav.classList.remove('open');
          toggle.setAttribute('aria-expanded', 'false');
        }
      });
    }
  }

  // ============================================================================
  // Scroll Animations
  // ============================================================================

  function initScrollAnimations() {
    const reveals = document.querySelectorAll('.reveal, .reveal-left, .reveal-right');

    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('active');
          }
        });
      }, { threshold: 0.1 });

      reveals.forEach(element => observer.observe(element));
    } else {
      // Fallback for browsers without IntersectionObserver
      reveals.forEach(element => element.classList.add('active'));
    }
  }

  // ============================================================================
  // Animated Counters
  // ============================================================================

  function initCounters() {
    const counters = document.querySelectorAll('[data-count]');

    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
            animateCounter(entry.target);
            entry.target.classList.add('counted');
          }
        });
      }, { threshold: 0.5 });

      counters.forEach(counter => observer.observe(counter));
    }
  }

  function animateCounter(element) {
    const target = parseFloat(element.getAttribute('data-count'));
    const duration = 2000;
    const increment = target / (duration / 16);
    let current = 0;

    const timer = setInterval(() => {
      current += increment;
      if (current >= target) {
        current = target;
        clearInterval(timer);
      }
      element.textContent = formatNumber(current, element.getAttribute('data-format'));
    }, 16);
  }

  function formatNumber(num, format) {
    if (format === 'percentage') {
      return num.toFixed(1) + '%';
    } else if (format === 'decimal') {
      return num.toFixed(2);
    } else {
      return Math.round(num).toLocaleString();
    }
  }

  // ============================================================================
  // Tabs
  // ============================================================================

  function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');

    tabButtons.forEach(button => {
      button.addEventListener('click', function() {
        const tabsContainer = this.closest('[data-tabs]');
        const targetId = this.getAttribute('data-tab');

        // Remove active class from all tabs
        tabsContainer.querySelectorAll('.tab-button').forEach(btn => {
          btn.classList.remove('active');
          btn.setAttribute('aria-selected', 'false');
        });

        tabsContainer.querySelectorAll('.tab-content').forEach(content => {
          content.classList.remove('active');
        });

        // Add active class to clicked tab
        this.classList.add('active');
        this.setAttribute('aria-selected', 'true');

        const targetContent = document.getElementById(targetId);
        if (targetContent) {
          targetContent.classList.add('active');
        }
      });
    });
  }

  // ============================================================================
  // Accordions
  // ============================================================================

  function initAccordions() {
    const accordionHeaders = document.querySelectorAll('.accordion-header');

    accordionHeaders.forEach(header => {
      header.addEventListener('click', function() {
        const item = this.parentElement;
        const body = item.querySelector('.accordion-body');
        const isActive = this.classList.contains('active');

        // Close all other accordions in the same container
        const accordion = item.closest('.accordion');
        accordion.querySelectorAll('.accordion-header').forEach(h => {
          h.classList.remove('active');
          h.setAttribute('aria-expanded', 'false');
        });
        accordion.querySelectorAll('.accordion-body').forEach(b => {
          b.classList.remove('active');
        });

        // Toggle current accordion
        if (!isActive) {
          this.classList.add('active');
          this.setAttribute('aria-expanded', 'true');
          body.classList.add('active');
        }
      });
    });
  }

  // ============================================================================
  // Code Blocks
  // ============================================================================

  function initCodeBlocks() {
    const copyButtons = document.querySelectorAll('.code-block-copy');

    copyButtons.forEach(button => {
      button.addEventListener('click', function() {
        const codeBlock = this.closest('.code-block');
        const code = codeBlock.querySelector('code').textContent;

        navigator.clipboard.writeText(code).then(() => {
          const originalText = this.textContent;
          this.textContent = 'Copied!';
          setTimeout(() => {
            this.textContent = originalText;
          }, 2000);
        }).catch(err => {
          console.error('Failed to copy:', err);
        });
      });
    });
  }

  // ============================================================================
  // Tooltips
  // ============================================================================

  function initTooltips() {
    const tooltips = document.querySelectorAll('[data-tooltip]');

    tooltips.forEach(element => {
      const text = element.getAttribute('data-tooltip');
      const tooltip = document.createElement('div');
      tooltip.className = 'tooltip';
      tooltip.textContent = text;

      const wrapper = document.createElement('div');
      wrapper.className = 'tooltip-wrapper';
      element.parentNode.insertBefore(wrapper, element);
      wrapper.appendChild(element);
      wrapper.appendChild(tooltip);
    });
  }

  // ============================================================================
  // Modals
  // ============================================================================

  function initModals() {
    const modalTriggers = document.querySelectorAll('[data-modal]');
    const modalCloses = document.querySelectorAll('.modal-close, .modal-overlay');

    modalTriggers.forEach(trigger => {
      trigger.addEventListener('click', function(e) {
        e.preventDefault();
        const modalId = this.getAttribute('data-modal');
        const modal = document.getElementById(modalId);
        if (modal) {
          modal.classList.add('active');
          document.body.style.overflow = 'hidden';
        }
      });
    });

    modalCloses.forEach(close => {
      close.addEventListener('click', function(e) {
        if (e.target === this) {
          const modal = this.closest('.modal-overlay');
          if (modal) {
            modal.classList.remove('active');
            document.body.style.overflow = '';
          }
        }
      });
    });

    // Close on escape
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        document.querySelectorAll('.modal-overlay.active').forEach(modal => {
          modal.classList.remove('active');
          document.body.style.overflow = '';
        });
      }
    });
  }

  // ============================================================================
  // Dropdowns
  // ============================================================================

  function initDropdowns() {
    const dropdownToggles = document.querySelectorAll('[data-dropdown]');

    dropdownToggles.forEach(toggle => {
      toggle.addEventListener('click', function(e) {
        e.stopPropagation();
        const dropdownId = this.getAttribute('data-dropdown');
        const dropdown = document.getElementById(dropdownId);

        // Close other dropdowns
        document.querySelectorAll('.dropdown').forEach(d => {
          if (d !== dropdown) d.classList.remove('open');
        });

        if (dropdown) {
          dropdown.classList.toggle('open');
        }
      });
    });

    // Close dropdowns on click outside
    document.addEventListener('click', function() {
      document.querySelectorAll('.dropdown.open').forEach(d => {
        d.classList.remove('open');
      });
    });
  }

  // ============================================================================
  // Back to Top Button
  // ============================================================================

  function initBackToTop() {
    const button = document.querySelector('.back-to-top');

    if (button) {
      window.addEventListener('scroll', function() {
        if (window.pageYOffset > 300) {
          button.classList.add('visible');
        } else {
          button.classList.remove('visible');
        }
      });

      button.addEventListener('click', function(e) {
        e.preventDefault();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }
  }

  // ============================================================================
  // Reading Progress
  // ============================================================================

  function initReadingProgress() {
    const progressBar = document.querySelector('.reading-progress');

    if (progressBar) {
      window.addEventListener('scroll', function() {
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const progress = (scrollTop / (documentHeight - windowHeight)) * 100;

        progressBar.style.width = Math.min(progress, 100) + '%';
      });
    }
  }

  // ============================================================================
  // Utility Functions
  // ============================================================================

  window.QuASIM = window.QuASIM || {};

  window.QuASIM.debounce = function(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  };

  window.QuASIM.throttle = function(func, limit) {
    let inThrottle;
    return function(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  };

  window.QuASIM.getJSON = async function(url) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching JSON:', error);
      return null;
    }
  };

  window.QuASIM.formatBytes = function(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

})();
