// Adaptive logo switcher for light/dark theme
(function() {
    'use strict';

    // Logo paths
    const LIGHT_LOGO = '_static/img/pyc_logo_transparent.png';
    const DARK_LOGO = '_static/img/pyc_logo_transparent_w.png';

    function updateLogos() {
        // Get current theme from data-theme attribute
        const theme = document.documentElement.getAttribute('data-theme');
        const isDark = theme === 'dark';

        // Update sidebar logo
        const sidebarLogo = document.querySelector('.sidebar-logo-img');
        if (sidebarLogo) {
            sidebarLogo.src = isDark ? DARK_LOGO : LIGHT_LOGO;
        }

        // Update any other logos with the adaptive class
        const adaptiveLogos = document.querySelectorAll('.adaptive-logo');
        adaptiveLogos.forEach(logo => {
            logo.src = isDark ? DARK_LOGO : LIGHT_LOGO;
        });
    }

    // Initial update
    updateLogos();

    // Watch for theme changes
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
                updateLogos();
            }
        });
    });

    // Start observing
    observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['data-theme']
    });

    // Also listen for theme toggle button clicks (backup method)
    document.addEventListener('click', function(e) {
        if (e.target.closest('.theme-toggle')) {
            setTimeout(updateLogos, 100);
        }
    });
})();

