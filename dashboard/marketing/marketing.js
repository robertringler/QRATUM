/**
 * QRATUM Marketing Pages JavaScript
 * Handles navigation, animations, cookie consent, and form handling
 * Version: 1.0.0
 */

(function() {
    'use strict';

    // ========================================
    // CONFIGURATION
    // ========================================
    const CONFIG = {
        cookieConsentKey: 'qratum_cookie_consent',
        animationThreshold: 0.1
    };

    // ========================================
    // UTILITY FUNCTIONS
    // ========================================
    const utils = {
        setCookie: function(name, value, days) {
            const expires = new Date();
            expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
            document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/;SameSite=Lax`;
        },

        getCookie: function(name) {
            const nameEQ = name + '=';
            const ca = document.cookie.split(';');
            for (let i = 0; i < ca.length; i++) {
                let c = ca[i];
                while (c.charAt(0) === ' ') c = c.substring(1, c.length);
                if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
            }
            return null;
        },

        debounce: function(func, wait) {
            let timeout;
            return function(...args) {
                clearTimeout(timeout);
                timeout = setTimeout(() => func.apply(this, args), wait);
            };
        }
    };

    // ========================================
    // MOBILE NAVIGATION
    // ========================================
    const mobileNav = {
        init: function() {
            const menuBtn = document.querySelector('.marketing-mobile-btn');
            const mobileNavEl = document.getElementById('mobile-nav');
            const closeBtn = document.querySelector('.marketing-mobile-nav__close');

            if (!menuBtn || !mobileNavEl) return;

            menuBtn.addEventListener('click', () => {
                mobileNavEl.classList.add('active');
                document.body.style.overflow = 'hidden';
                menuBtn.setAttribute('aria-expanded', 'true');
            });

            const closeMobileNav = () => {
                mobileNavEl.classList.remove('active');
                document.body.style.overflow = '';
                menuBtn.setAttribute('aria-expanded', 'false');
            };

            if (closeBtn) {
                closeBtn.addEventListener('click', closeMobileNav);
            }

            // Close on link click
            mobileNavEl.querySelectorAll('a').forEach(link => {
                link.addEventListener('click', closeMobileNav);
            });

            // Close on escape key
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && mobileNavEl.classList.contains('active')) {
                    closeMobileNav();
                }
            });

            // Close on click outside
            mobileNavEl.addEventListener('click', (e) => {
                if (e.target === mobileNavEl) {
                    closeMobileNav();
                }
            });
        }
    };

    // ========================================
    // HEADER SCROLL BEHAVIOR
    // ========================================
    const headerScroll = {
        init: function() {
            const header = document.querySelector('.marketing-header');
            if (!header) return;

            const handleScroll = utils.debounce(() => {
                if (window.pageYOffset > 50) {
                    header.classList.add('scrolled');
                } else {
                    header.classList.remove('scrolled');
                }
            }, 10);

            window.addEventListener('scroll', handleScroll, { passive: true });
        }
    };

    // ========================================
    // SMOOTH SCROLL
    // ========================================
    const smoothScroll = {
        init: function() {
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function(e) {
                    const href = this.getAttribute('href');
                    if (href === '#') return;

                    const target = document.querySelector(href);
                    if (target) {
                        e.preventDefault();
                        const headerHeight = 80;
                        const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - headerHeight;
                        window.scrollTo({
                            top: targetPosition,
                            behavior: 'smooth'
                        });
                    }
                });
            });
        }
    };

    // ========================================
    // SCROLL ANIMATIONS
    // ========================================
    const scrollAnimations = {
        init: function() {
            const elements = document.querySelectorAll('.feature-card, .usecase-card');
            if (elements.length === 0) return;

            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }
                });
            }, {
                threshold: CONFIG.animationThreshold,
                rootMargin: '0px 0px -50px 0px'
            });

            elements.forEach(el => {
                el.style.opacity = '0';
                el.style.transform = 'translateY(20px)';
                el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
                observer.observe(el);
            });
        }
    };

    // ========================================
    // COOKIE CONSENT
    // ========================================
    const cookieConsent = {
        init: function() {
            const banner = document.getElementById('cookie-banner');
            if (!banner) return;

            // Check if user has already consented
            const consent = utils.getCookie(CONFIG.cookieConsentKey);
            if (consent) return;

            // Show banner after short delay
            setTimeout(() => {
                banner.classList.add('visible');
            }, 1000);

            // Accept button
            const acceptBtn = document.getElementById('cookie-accept');
            if (acceptBtn) {
                acceptBtn.addEventListener('click', () => {
                    utils.setCookie(CONFIG.cookieConsentKey, 'accepted', 365);
                    banner.classList.remove('visible');
                    console.log('[QRATUM] Cookie consent accepted');
                });
            }

            // Reject button
            const rejectBtn = document.getElementById('cookie-reject');
            if (rejectBtn) {
                rejectBtn.addEventListener('click', () => {
                    utils.setCookie(CONFIG.cookieConsentKey, 'rejected', 365);
                    banner.classList.remove('visible');
                    console.log('[QRATUM] Cookie consent rejected');
                });
            }
        }
    };

    // ========================================
    // COUNTER ANIMATION
    // ========================================
    const counterAnimation = {
        init: function() {
            const metrics = document.querySelectorAll('.hero__metric-value, .stat__value');
            if (metrics.length === 0) return;

            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        this.animate(entry.target);
                        observer.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.5 });

            metrics.forEach(metric => observer.observe(metric));
        },

        animate: function(element) {
            const text = element.textContent;
            const match = text.match(/^([<>]?)(\d+\.?\d*)(.*)$/);
            
            if (!match) return;
            
            const prefix = match[1] || '';
            const target = parseFloat(match[2]);
            const suffix = match[3] || '';
            const decimals = match[2].includes('.') ? 1 : 0;
            
            if (isNaN(target)) return;
            
            const duration = 2000;
            const startTime = performance.now();
            
            const updateCounter = (currentTime) => {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const easeOut = 1 - Math.pow(1 - progress, 3);
                const current = target * easeOut;
                
                element.textContent = prefix + current.toFixed(decimals) + suffix;
                
                if (progress < 1) {
                    requestAnimationFrame(updateCounter);
                }
            };
            
            requestAnimationFrame(updateCounter);
        }
    };

    // ========================================
    // FORM HANDLING
    // ========================================
    const formHandler = {
        init: function() {
            const forms = document.querySelectorAll('form[data-form]');
            forms.forEach(form => {
                form.addEventListener('submit', this.handleSubmit.bind(this));
            });
        },

        handleSubmit: async function(e) {
            e.preventDefault();
            const form = e.target;
            const submitBtn = form.querySelector('[type="submit"]');
            const originalText = submitBtn?.textContent || 'Submit';

            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.textContent = 'Submitting...';
            }

            try {
                // Simulate form submission
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                this.showMessage(form, 'success', 'Thank you! We\'ll be in touch soon.');
                form.reset();
            } catch (error) {
                this.showMessage(form, 'error', 'Something went wrong. Please try again.');
            } finally {
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.textContent = originalText;
                }
            }
        },

        showMessage: function(form, type, message) {
            const existingMsg = form.querySelector('.form-message');
            if (existingMsg) existingMsg.remove();

            const msgEl = document.createElement('div');
            msgEl.className = `form-message form-message--${type}`;
            msgEl.textContent = message;
            msgEl.style.cssText = `
                padding: 12px 16px;
                margin-top: 16px;
                border-radius: 6px;
                background: ${type === 'success' ? 'rgba(0, 255, 136, 0.1)' : 'rgba(255, 51, 102, 0.1)'};
                color: ${type === 'success' ? 'var(--success)' : 'var(--error)'};
                border: 1px solid ${type === 'success' ? 'rgba(0, 255, 136, 0.3)' : 'rgba(255, 51, 102, 0.3)'};
            `;
            
            form.appendChild(msgEl);
            
            setTimeout(() => msgEl.remove(), 5000);
        }
    };

    // ========================================
    // PARTICLE ANIMATION (Hero Background)
    // ========================================
    const particleAnimation = {
        canvas: null,
        ctx: null,
        particles: [],
        animationId: null,
        resizeHandler: null,

        init: function() {
            const hero = document.querySelector('.hero');
            if (!hero) return;

            // Create canvas
            this.canvas = document.createElement('canvas');
            this.canvas.style.position = 'absolute';
            this.canvas.style.top = '0';
            this.canvas.style.left = '0';
            this.canvas.style.width = '100%';
            this.canvas.style.height = '100%';
            this.canvas.style.pointerEvents = 'none';
            this.canvas.style.opacity = '0.5';
            
            const heroBackground = hero.querySelector('.hero__background');
            if (heroBackground) {
                heroBackground.insertBefore(this.canvas, heroBackground.firstChild);
            }

            this.ctx = this.canvas.getContext('2d');
            this.resize();
            this.createParticles();
            this.animate();

            // Handle resize with cleanup reference
            this.resizeHandler = utils.debounce(() => {
                this.resize();
                this.createParticles();
            }, 250);
            window.addEventListener('resize', this.resizeHandler);
        },

        resize: function() {
            this.canvas.width = this.canvas.offsetWidth;
            this.canvas.height = this.canvas.offsetHeight;
        },

        createParticles: function() {
            // Configurable particle density (can be overridden via data attribute)
            const density = parseInt(this.canvas.dataset?.particleDensity || '30', 10);
            const count = Math.min(50, Math.floor(this.canvas.width / density));
            this.particles = [];

            for (let i = 0; i < count; i++) {
                this.particles.push({
                    x: Math.random() * this.canvas.width,
                    y: Math.random() * this.canvas.height,
                    size: Math.random() * 3 + 1,
                    speedX: (Math.random() - 0.5) * 0.5,
                    speedY: (Math.random() - 0.5) * 0.5,
                    opacity: Math.random() * 0.5 + 0.2
                });
            }
        },

        animate: function() {
            if (!this.ctx) return;

            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

            // Update and draw particles
            this.particles.forEach((particle, index) => {
                // Update position
                particle.x += particle.speedX;
                particle.y += particle.speedY;

                // Wrap around edges
                if (particle.x < 0) particle.x = this.canvas.width;
                if (particle.x > this.canvas.width) particle.x = 0;
                if (particle.y < 0) particle.y = this.canvas.height;
                if (particle.y > this.canvas.height) particle.y = 0;

                // Draw particle
                this.ctx.beginPath();
                this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
                this.ctx.fillStyle = `rgba(139, 92, 246, ${particle.opacity})`;
                this.ctx.fill();

                // Draw connections
                for (let j = index + 1; j < this.particles.length; j++) {
                    const other = this.particles[j];
                    const dx = particle.x - other.x;
                    const dy = particle.y - other.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < 120) {
                        this.ctx.beginPath();
                        this.ctx.moveTo(particle.x, particle.y);
                        this.ctx.lineTo(other.x, other.y);
                        this.ctx.strokeStyle = `rgba(139, 92, 246, ${0.15 * (1 - distance / 120)})`;
                        this.ctx.lineWidth = 1;
                        this.ctx.stroke();
                    }
                }
            });

            this.animationId = requestAnimationFrame(this.animate.bind(this));
        },

        destroy: function() {
            if (this.animationId) {
                cancelAnimationFrame(this.animationId);
                this.animationId = null;
            }
            if (this.resizeHandler) {
                window.removeEventListener('resize', this.resizeHandler);
                this.resizeHandler = null;
            }
            if (this.canvas && this.canvas.parentNode) {
                this.canvas.parentNode.removeChild(this.canvas);
            }
            this.canvas = null;
            this.ctx = null;
            this.particles = [];
        }
    };

    // ========================================
    // INITIALIZATION
    // ========================================
    const init = function() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initModules);
        } else {
            initModules();
        }
    };

    const initModules = function() {
        mobileNav.init();
        headerScroll.init();
        smoothScroll.init();
        scrollAnimations.init();
        cookieConsent.init();
        counterAnimation.init();
        formHandler.init();
        particleAnimation.init();

        console.log('[QRATUM] Marketing page initialized');
    };

    // Start initialization
    init();

    // Expose public API
    window.QRATUMMarketing = {
        showCookieBanner: () => {
            const banner = document.getElementById('cookie-banner');
            if (banner) banner.classList.add('visible');
        }
    };

})();
