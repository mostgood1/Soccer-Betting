// All Matches (by date across leagues)
class AllMatchesManager {
  constructor(apiBaseUrl) {
    this.apiBaseUrl = apiBaseUrl;
    this.groups = [];
    this.brandingAll = {};
    this.oddsByLeagueWeek = {}; // { `${league}-${week}`: [ {home_team, away_team, odds} ] }
    try {
      const url = new URL(window.location.href);
      this.filterLeague = url.searchParams.get('league') || 'ALL';
      this.daysAhead = parseInt(url.searchParams.get('days') || '14', 10);
    } catch {
      this.filterLeague = 'ALL';
      this.daysAhead = 14;
    }
    this.init();
  }

  async init() {
    // Load static data first so we can render quickly
    try {
      await Promise.all([
        this.loadBrandingAll(),
        this.loadGroups()
      ]);
    } catch (e) {
      console.warn('Failed initial loads for AllMatches; proceeding with empty groups', e);
      if (!Array.isArray(this.groups)) this.groups = [];
    }
    // Render immediately; do not block on network odds
    this.render();
    this.setupNavHandlers();
    // Kick off odds loading in the background; re-render when done
    this.loadBatchOdds()
      .then(() => {
        try { this.render(); } catch (_) {}
      })
      .catch(() => {/* ignore odds failures for initial UX */});
  }

  async loadBrandingAll() {
    try {
      const resp = await fetch(`${this.apiBaseUrl}/api/branding/teams/all`);
      if (!resp.ok) throw new Error('branding all failed');
      const data = await resp.json();
      this.brandingAll = data.branding || {};
    } catch (e) {
      console.warn('branding (all) unavailable', e);
      this.brandingAll = {};
    }
  }

  async loadGroups() {
    try {
      const qs = [];
      if (this.filterLeague && this.filterLeague !== 'ALL') qs.push(`leagues=${encodeURIComponent(this.filterLeague)}`);
  qs.push(`days_ahead=${encodeURIComponent(this.daysAhead)}`);
  qs.push(`days_back=7`);
  qs.push(`include_completed=true`);
      const url = `${this.apiBaseUrl}/api/games/by-date?${qs.join('&')}`;
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`by-date fetch failed (${resp.status})`);
      const data = await resp.json();
      this.groups = Array.isArray(data.groups) ? data.groups : [];
    } catch (e) {
      console.warn('by-date groups unavailable', e);
      this.groups = [];
    }
  }

  async loadBatchOdds() {
    // Batch market odds lookups by (league, week)
    const requests = [];
    const seen = new Set();
    for (const g of this.groups) {
      for (const m of g.matches || []) {
        const key = `${m.league}-${m.game_week}`;
        if (!seen.has(key)) {
          seen.add(key);
          requests.push(this._fetchWeekOdds(m.league, m.game_week));
        }
      }
    }
    await Promise.allSettled(requests);
  }

  async _fetchWeekOdds(league, week) {
    try {
      // Add a fetch timeout to avoid hanging the UI if providers are slow
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 10000);
      const resp = await fetch(
        `${this.apiBaseUrl}/api/betting/odds/week/${week}?limit=20&league=${encodeURIComponent(league)}`,
        { signal: controller.signal }
      );
      clearTimeout(t);
      if (!resp.ok) throw new Error(`week odds failed ${league}-${week}`);
      const data = await resp.json();
      this.oddsByLeagueWeek[`${league}-${week}`] = data.matches || [];
    } catch (e) {
      console.warn('week odds error', league, week, e);
      this.oddsByLeagueWeek[`${league}-${week}`] = [];
    }
  }

  getOddsRow(league, week, home, away) {
    const list = this.oddsByLeagueWeek[`${league}-${week}`] || [];
    return list.find(r => r.home_team === home && r.away_team === away) || null;
  }

  getTeamLogo(name) {
    if (!name) return '/static/placeholder.png';
    const rec = this.brandingAll[name];
    return (rec && rec.crest) ? rec.crest : '/static/placeholder.png';
  }

  fmtAmerican(odd) {
    if (odd === null || odd === undefined) return '—';
    const n = Number(odd);
    if (!isFinite(n) || n === 0) return '—';
    return (n > 0 ? '+' : '') + Math.round(n);
  }

  formatLocalDateParts(isoLike) {
    try {
      const d = isoLike ? new Date(isoLike) : null;
      if (!d || isNaN(d.getTime())) return { date: '—', time: '—', long: '—' };
      const locale = (navigator && navigator.language) ? navigator.language : 'en-US';
      const dateFmt = new Intl.DateTimeFormat(locale, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
      const timeFmt = new Intl.DateTimeFormat(locale, { hour: 'numeric', minute: '2-digit', timeZoneName: 'short' });
      return { date: dateFmt.format(d), time: timeFmt.format(d), long: dateFmt.format(d) + ' ' + timeFmt.format(d) };
    } catch (e) { return { date: '—', time: '—', long: '—' }; }
  }

  render() {
    const container = document.getElementById('all-matches-container');
    if (!container) return;
    if (!this.groups.length) {
      container.innerHTML = '<div class="no-matches">No upcoming matches found</div>';
      return;
    }
    const html = this.groups.map(g => this.renderGroup(g)).join('');
    container.innerHTML = html;
  }

  renderGroup(group) {
    const dparts = this.formatLocalDateParts(group.date);
    const cards = (group.matches || []).map(m => this.renderCard(m)).join('');
    return `
      <div class="day-group">
        <h3 class="day-header">${dparts.date}</h3>
        <div class="game-cards-grid">${cards}</div>
      </div>`;
  }

  renderCard(m) {
    const status = m.is_completed ? 'completed' : ((m.status||'').toUpperCase()==='IN_PLAY' ? 'live' : 'upcoming');
    const kickoff = m.utc_date || m.date;
    const kp = this.formatLocalDateParts(kickoff);
    const oddsRow = this.getOddsRow(m.league, m.game_week, m.home_team, m.away_team);
    let oddsText = 'Odds: —';
    try {
      if (oddsRow && oddsRow.odds && oddsRow.odds.market_odds) {
        const mw = oddsRow.odds.market_odds.match_winner || {};
        const h = this.fmtAmerican(mw.home?.odds_american);
        const d = this.fmtAmerican(mw.draw?.odds_american);
        const a = this.fmtAmerican(mw.away?.odds_american);
        oddsText = `Odds: H ${h} | D ${d} | A ${a}`;
      }
    } catch {}
    const leagueBadge = `<span class="sb-chip"><i class="fas fa-flag"></i> ${m.league}</span>`;
    const venue = m.venue || m.stadium || '';
    return `
      <div class="game-card sb-card ${status}">
        <div class="sb-card-top">
          <div class="sb-meta-left"><span class="sb-date" title="${kp.long}">${kp.date} • ${kp.time}</span></div>
          <div class="sb-status badge-${status}">${status.charAt(0).toUpperCase()+status.slice(1)} ${leagueBadge}</div>
        </div>
        <div class="sb-meta-bottom">
          ${venue ? `<span class="sb-chip"><i class="fas fa-location-dot"></i> ${venue}</span>` : ''}
          <span class="sb-chip sb-market-odds">${oddsText}</span>
        </div>
        <div class="sb-match-row">
          <div class="sb-team away vertical">
            <div class="logo-wrap lg"><img src="${this.getTeamLogo(m.away_team)}" alt="${m.away_team}" /></div>
            <span class="t-name">${m.away_team}</span>
          </div>
          <div class="sb-center">${m.is_completed ? `<div class='final-score'>${m.away_score||0}<span class='dash'>-</span>${m.home_score||0}</div>` : '<div class="vs-pill">VS</div>'}</div>
          <div class="sb-team home vertical">
            <div class="logo-wrap lg"><img src="${this.getTeamLogo(m.home_team)}" alt="${m.home_team}" /></div>
            <span class="t-name">${m.home_team}</span>
          </div>
        </div>
        <div class="sb-actions">
          <a class="sb-link" href="/?league=${encodeURIComponent(m.league)}&week=${encodeURIComponent(m.game_week)}" title="Open week view">Week ${m.game_week} details</a>
        </div>
      </div>`;
  }

  setupNavHandlers() {
    try {
      const sel = document.getElementById('league-select');
      if (sel) {
        sel.addEventListener('change', () => {
          const url = new URL(window.location.href);
          if (sel.value === 'ALL') url.searchParams.delete('league'); else url.searchParams.set('league', sel.value);
          window.location.href = url.toString();
        });
      }
    } catch {}
  }
}

window.AllMatchesManager = AllMatchesManager;
