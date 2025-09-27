// Minimal frontend bootstrap for Game Weeks only
class EPLBettingApp {
  constructor() {
    this.apiBaseUrl = (typeof window !== 'undefined' && window.API_BASE_URL)
      ? window.API_BASE_URL
      : `${window.location.protocol}//${window.location.host}`;
    this.gameWeekManager = null;
    this.allMatchesManager = null;
    this.init();
  }

  async init() {
    try {
      // Initialize All Matches (default landing)
      this.allMatchesManager = new AllMatchesManager(this.apiBaseUrl);
      // Initialize GameWeekManager too for quick tab switch; honor ?week= override
      const url = new URL(window.location.href);
      const weekParam = parseInt(url.searchParams.get('week') || '0', 10);
      this.gameWeekManager = new GameWeekManager(this.apiBaseUrl);
      if (weekParam && !isNaN(weekParam) && weekParam >= 1 && weekParam <= 38) {
        // Defer to allow initial load
        setTimeout(()=> this.gameWeekManager.setWeek(weekParam), 300);
      }
    } catch (e) {
      console.error('Failed to initialize GameWeekManager:', e);
    }
    // Theme
    document.body.classList.add('dark-sb');
  }
}

let app;
document.addEventListener('DOMContentLoaded', () => {
  app = new EPLBettingApp();
});