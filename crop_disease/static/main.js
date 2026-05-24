// Add loading animation on form submit
document.querySelector('form')?.addEventListener('submit', function() {
    const button = this.querySelector('button[type="submit"]');
    if (button) {
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span> Analyzing...';
    }
});