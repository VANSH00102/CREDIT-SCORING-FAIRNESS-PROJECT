/* CreditFairAI — Global JS */

document.addEventListener('DOMContentLoaded', () => {
  // Active nav link
  const path = window.location.pathname;
  document.querySelectorAll('.navbar .nav-link').forEach(a => {
    if (a.getAttribute('href') === path) a.classList.add('active');
  });

  // Auto-dismiss success/info alerts after 6 s
  document.querySelectorAll('.alert.alert-success,.alert.alert-info').forEach(el => {
    setTimeout(() => {
      if (el && el.parentNode) {
        el.style.transition = 'opacity .5s';
        el.style.opacity = '0';
        setTimeout(() => el.remove(), 500);
      }
    }, 6000);
  });
});

function fmt(n, d=4){ return n==null?'—':parseFloat(n).toFixed(d); }
