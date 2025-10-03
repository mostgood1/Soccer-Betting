// All Matches (by date across leagues)
class AllMatchesManager {
  constructor(apiBaseUrl) {
    this.apiBaseUrl = apiBaseUrl;
    this.groups = [];
    this.brandingAll = {};
    this.oddsByLeagueWeek = {}; // { `${league}-${week}`: [ {home_team, away_team, odds} ] }
    this.predsByLeagueWeek = {}; // { `${league}-${week}`: [ {home_team, away_team, predictions} ] }
    try {
      const url = new URL(window.location.href);
      this.filterLeague = url.searchParams.get('league') || 'ALL';
      // All Matches default: current day only (can override via ?days=N)
      const daysParam = url.searchParams.get('days');
      const parsed = parseInt(daysParam || '', 10);
      this.onlyToday = (url.searchParams.get('today') || '') !== '0';
      // Backend requires days_ahead >= 1; use 1 and filter to today client-side when onlyToday
      this.daysAhead = Number.isFinite(parsed) ? Math.max(1, parsed) : 1;
      this.daysBack = 0;
    } catch {
      this.filterLeague = 'ALL';
      this.onlyToday = true;
      this.daysAhead = 1; // satisfy API
      this.daysBack = 0;
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
    // Kick off predictions fetch (compact) and re-render when available
    this.loadBatchPredictions()
      .then(() => { try { this.render(); } catch (_) {} })
      .catch(() => {/* ignore */});
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
  qs.push(`days_back=${encodeURIComponent(this.daysBack)}`);
  // Show completed games from the same day as well (like NHL view)
  qs.push(`include_completed=true`);
      const url = `${this.apiBaseUrl}/api/games/by-date?${qs.join('&')}`;
      // Guard against slow API: add a 10s timeout
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 10000);
      const resp = await fetch(url, { signal: controller.signal });
      clearTimeout(t);
      if (!resp.ok) throw new Error(`by-date fetch failed (${resp.status})`);
      const data = await resp.json();
      let groups = Array.isArray(data.groups) ? data.groups : [];
      if (this.onlyToday && groups.length) {
        const todayUtc = new Date().toISOString().slice(0,10);
        // First, restrict to the server's "today" grouping by UTC
        groups = groups.filter(g => (g.date || '').slice(0,10) === todayUtc);
        // Then, additionally prune each group's matches to the viewer's LOCAL calendar day
        // so that the Friday-only view doesn't include local Saturday kickoffs.
        const todayLocalKey = this._localDateKeyFromDate(new Date());
        groups = this._pruneGroupsToLocalDay(groups, todayLocalKey);
      }
      // UX fallback: if no matches today, auto-show the nearest upcoming match day within 7 days
      if ((!groups || groups.length === 0) && this.onlyToday) {
        try {
          const url2 = `${this.apiBaseUrl}/api/games/by-date?days_ahead=7&days_back=0&include_completed=false${this.filterLeague && this.filterLeague !== 'ALL' ? `&leagues=${encodeURIComponent(this.filterLeague)}` : ''}`;
          const r2 = await fetch(url2);
          if (r2.ok) {
            const d2 = await r2.json();
            const g2 = Array.isArray(d2.groups) ? d2.groups : [];
            if (g2.length) {
              // pick earliest date
              g2.sort((a,b) => (a.date||'').localeCompare(b.date||''));
              let nextGroup = { ...g2[0] };
              // Determine target local date key based on earliest match's local day if available
              const firstMatch = (nextGroup.matches || [])[0] || null;
              const targetLocalKey = firstMatch ? this._localDateKeyFromISO(firstMatch.utc_date || firstMatch.date) : this._localDateKeyFromISO(nextGroup.date);
              // Prune the group's matches to the target local day
              nextGroup.matches = (nextGroup.matches || []).filter(m => this._localDateKeyFromISO(m.utc_date || m.date) === targetLocalKey);
              // If pruning removed everything (edge case), fall back to original matches
              if (!nextGroup.matches.length) nextGroup = { ...g2[0] };
              // Align the group's displayed date with the local day we selected
              nextGroup.date = this._isoFromLocalKey(targetLocalKey) || nextGroup.date;
              this.groups = [nextGroup];
              // Use a local-facing date for the banner
              this.fallbackInfo = { type: 'next-day', date: this._isoFromLocalKey(targetLocalKey) };
              return;
            }
          }
        } catch (_) { /* ignore */ }
      }
      this.groups = groups;
    } catch (e) {
      console.warn('by-date groups unavailable', e);
      // Try a per-league fallback to avoid a single slow league blocking the page
      try {
        const leagues = this.filterLeague && this.filterLeague !== 'ALL' ? [this.filterLeague] : ['PL','BL1','FL1','SA','PD'];
        const perGroups = await this.loadGroupsPerLeague(leagues);
        if (perGroups && perGroups.length) {
          // When showing only today, prune to local day too
          this.groups = this.onlyToday ? this._pruneGroupsToLocalDay(perGroups, this._localDateKeyFromDate(new Date())) : perGroups;
          return;
        }
      } catch (e2) {
        console.warn('per-league fallback failed', e2);
      }
      // Final fallback: show a banner so the UI isn’t stuck
      this.groups = [];
      this.fallbackInfo = { type: 'error', message: 'API timeout or unavailable' };
    }
  }

  async loadGroupsPerLeague(leagues) {
    // Fetch per-league by-date concurrently with short timeouts and merge by date
    const results = await Promise.allSettled(leagues.map(lg => this._fetchGroupsForLeague(lg)));
    const merged = {};
    for (const r of results) {
      if (r.status !== 'fulfilled' || !r.value) continue;
      for (const g of r.value) {
        const key = g.date;
        const list = merged[key] || [];
        merged[key] = list.concat(g.matches || []);
      }
    }
    const out = Object.keys(merged).sort().map(d => ({ date: d, matches: merged[d] }));
    // Apply onlyToday filter
    if (this.onlyToday && out.length) {
      const todayUtc = new Date().toISOString().slice(0,10);
      return out.filter(g => (g.date || '').slice(0,10) === todayUtc);
    }
    return out;
  }

  async _fetchGroupsForLeague(league) {
    try {
      const qs = [
        `leagues=${encodeURIComponent(league)}`,
        `days_ahead=${encodeURIComponent(this.daysAhead)}`,
        `days_back=${encodeURIComponent(this.daysBack)}`,
        `include_completed=true`
      ];
      const url = `${this.apiBaseUrl}/api/games/by-date?${qs.join('&')}`;
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 8000);
      const resp = await fetch(url, { signal: controller.signal });
      clearTimeout(t);
      if (!resp.ok) return [];
      const data = await resp.json();
      return Array.isArray(data.groups) ? data.groups : [];
    } catch {
      return [];
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

  async loadBatchPredictions() {
    const reqs = [];
    const seen = new Set();
    for (const g of this.groups) {
      for (const m of g.matches || []) {
        const key = `${m.league}-${m.game_week}`;
        if (!seen.has(key)) {
          seen.add(key);
          reqs.push(this._fetchWeekPredictions(m.league, m.game_week));
        }
      }
    }
    await Promise.allSettled(reqs);
  }

  async _fetchWeekOdds(league, week) {
    try {
      // Add a fetch timeout to avoid hanging the UI if providers are slow
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 10000);
      // Prefer live provider calls here so All Matches shows complete markets (Totals/BTTS/Corners)
      // within a bounded timeout. If this fails, we retry once with cache_only=true.
      let resp = await fetch(
        `${this.apiBaseUrl}/api/betting/odds/week/${week}?limit=20&league=${encodeURIComponent(league)}&cache_only=false`,
        { signal: controller.signal }
      );
      clearTimeout(t);
      if (!resp.ok) {
        // Fallback to cache_only when live call is slow/unavailable
        const controller2 = new AbortController();
        const t2 = setTimeout(() => controller2.abort(), 8000);
        resp = await fetch(
          `${this.apiBaseUrl}/api/betting/odds/week/${week}?limit=20&league=${encodeURIComponent(league)}&cache_only=true`,
          { signal: controller2.signal }
        );
        clearTimeout(t2);
        if (!resp.ok) throw new Error(`week odds failed ${league}-${week}`);
      }
      const data = await resp.json();
      this.oddsByLeagueWeek[`${league}-${week}`] = data.matches || [];
    } catch (e) {
      console.warn('week odds error', league, week, e);
      this.oddsByLeagueWeek[`${league}-${week}`] = [];
    }
  }

  async _fetchWeekPredictions(league, week) {
    try {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 10000);
      const resp = await fetch(
        `${this.apiBaseUrl}/api/predictions/week/${encodeURIComponent(week)}?league=${encodeURIComponent(league)}&allow_on_demand=0`,
        { signal: controller.signal }
      );
      clearTimeout(t);
      if (!resp.ok) throw new Error(`week preds failed ${league}-${week}`);
      const data = await resp.json();
      this.predsByLeagueWeek[`${league}-${week}`] = data.matches || [];
    } catch (e) {
      console.warn('week predictions error', league, week, e);
      this.predsByLeagueWeek[`${league}-${week}`] = [];
    }
  }

  getOddsRow(league, week, home, away) {
    const list = this.oddsByLeagueWeek[`${league}-${week}`] || [];
    return list.find(r => r.home_team === home && r.away_team === away) || null;
  }

  getPredRow(league, week, home, away) {
    const list = this.predsByLeagueWeek[`${league}-${week}`] || [];
    return list.find(r => r.home_team === home && r.away_team === away) || null;
  }

  getTeamLogo(name) {
    if (!name) return '/placeholder.png';
    // Exact hit first
    const rec = this.brandingAll[name];
    if (rec && rec.crest) return rec.crest;
    // Common aliases across leagues
    const alias = {
      // PL
      'Man City': 'Manchester City FC',
      'Manchester City': 'Manchester City FC',
      'Man United': 'Manchester United FC',
      'Manchester United': 'Manchester United FC',
      'Tottenham': 'Tottenham Hotspur FC',
      'West Ham': 'West Ham United FC',
      'Newcastle': 'Newcastle United FC',
      'Leeds': 'Leeds United FC',
      'Leeds United': 'Leeds United FC',
      'Bournemouth': 'AFC Bournemouth',
      'Fulham': 'Fulham FC',
      'Everton': 'Everton FC',
      'Brighton': 'Brighton & Hove Albion FC',
      'Nottm Forest': 'Nottingham Forest FC',
      'Nottingham Forest': 'Nottingham Forest FC',
      'Wolves': 'Wolverhampton Wanderers FC',
      'Brentford': 'Brentford FC',
      'Leicester': 'Leicester City FC',
      'Southampton': 'Southampton FC',
      'Ipswich': 'Ipswich Town FC',
      'Sunderland': 'Sunderland AFC',
      'Sheffield United': 'Sheffield United FC',
      'Sheffield Utd': 'Sheffield United FC',
      'Sheffield Wednesday': 'Sheffield Wednesday FC',
      // BL1
      'Bayern': 'FC Bayern München',
      'Leverkusen': 'Bayer 04 Leverkusen',
      'Dortmund': 'Borussia Dortmund',
      'Gladbach': 'Borussia Mönchengladbach',
      'Mönchengladbach': 'Borussia Mönchengladbach',
      'Monchengladbach': 'Borussia Mönchengladbach',
      'Köln': '1. FC Köln',
      'Koln': '1. FC Köln',
      'Frankfurt': 'Eintracht Frankfurt',
      'Hoffenheim': 'TSG 1899 Hoffenheim',
      'Stuttgart': 'VfB Stuttgart',
      'Wolfsburg': 'VfL Wolfsburg',
      'Leipzig': 'RB Leipzig',
      'Bochum': 'VfL Bochum 1848',
      'Heidenheim': '1. FC Heidenheim 1846',
      'Union Berlin': '1. FC Union Berlin',
      'Augsburg': 'FC Augsburg',
      'Mainz': '1. FSV Mainz 05',
      'Bremen': 'SV Werder Bremen',
      // La Liga
      'Atletico Madrid': 'Atlético de Madrid',
      'Athletic Bilbao': 'Athletic Club',
      'Real Betis': 'Real Betis Balompié',
      'Celta Vigo': 'RC Celta de Vigo',
      'Deportivo Alaves': 'Deportivo Alavés',
      'Real Sociedad': 'Real Sociedad de Fútbol',
      'Mallorca': 'RCD Mallorca',
      // Serie A
      'Inter': 'FC Internazionale Milano',
      'AC Milan': 'AC Milan',
      'AS Roma': 'AS Roma',
      'Lazio': 'SS Lazio',
      'Napoli': 'SSC Napoli',
      'Fiorentina': 'ACF Fiorentina',
      'Verona': 'Hellas Verona FC',
      'Udinese': 'Udinese Calcio',
      'Monza': 'AC Monza'
    };
    const full = alias[name];
    if (full && this.brandingAll[full]?.crest) return this.brandingAll[full].crest;
    // Case-insensitive direct match
    const lower = name.toLowerCase();
    const direct = Object.keys(this.brandingAll).find(k => k.toLowerCase() === lower);
    if (direct && this.brandingAll[direct]?.crest) return this.brandingAll[direct].crest;
    // Normalized fuzzy match
    const norm = (s) => (s || '')
      .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
      .replace(/&/g,' and ')
      .replace(/[^a-z0-9 ]/gi,' ')
      .replace(/\b(f\.?c\.?|a\.?f\.?c\.?|c\.?f\.?|s\.?c\.?)\b/gi,'')
      .replace(/\s+/g,' ')
      .trim()
      .toLowerCase();
    const keyN = norm(name);
    const hit = Object.keys(this.brandingAll).find(k => norm(k) === keyN);
    if (hit && this.brandingAll[hit]?.crest) return this.brandingAll[hit].crest;
    const contains = Object.keys(this.brandingAll).find(k => norm(k).includes(keyN) || keyN.includes(norm(k)));
    if (contains && this.brandingAll[contains]?.crest) return this.brandingAll[contains].crest;
    return '/placeholder.png';
  }

  fmtAmerican(odd) {
    if (odd === null || odd === undefined) return '—';
    const n = Number(odd);
    if (!isFinite(n) || n === 0) return '—';
    return (n > 0 ? '+' : '') + Math.round(n);
  }

  formatLocalDateParts(isoLike) {
    try {
      let src = isoLike;
      // If the string looks like an ISO without timezone (naive), treat it as UTC to avoid local-time drift
      if (typeof src === 'string' && /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(src) && !(/[zZ]|[\+\-]\d{2}:?\d{2}$/.test(src))) {
        src = src + 'Z';
      }
      const d = src ? new Date(src) : null;
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
    const banner = this.renderBanner();
    const html = (banner ? banner : '') + this.groups.map(g => this.renderGroup(g)).join('');
    container.innerHTML = html;
  }

  renderBanner() {
    try {
      if (!this.fallbackInfo) return '';
      const dparts = this.formatLocalDateParts(this.fallbackInfo.date);
      if (this.fallbackInfo.type === 'error') {
        return `<div class="sb-info-banner">Unable to load matches right now (timeout). You can refresh or try again shortly.</div>`;
      }
      return `<div class="sb-info-banner">No matches today — showing next match day: <strong>${dparts.date}</strong></div>`;
    } catch { return ''; }
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
  const marketsHtml = this.renderMarkets(oddsRow?.odds);
  const recHtml = this.renderRecommendation(m, oddsRow);
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
  ${recHtml}
  ${marketsHtml}
        <div class="sb-actions">
          <a class="sb-link" href="/?league=${encodeURIComponent(m.league)}&week=${encodeURIComponent(m.game_week)}" title="Open week view">Week ${m.game_week} details</a>
        </div>
      </div>`;
  }

  renderMarkets(odds) {
    try {
      if (!odds || !odds.market_odds) return '';
      const mo = odds.market_odds;
      // Moneyline row
      let ml = '';
      if (mo.match_winner) {
        const h = this.fmtAmerican(mo.match_winner.home?.odds_american);
        const d = this.fmtAmerican(mo.match_winner.draw?.odds_american);
        const a = this.fmtAmerican(mo.match_winner.away?.odds_american);
        ml = `
          <div class="sb-market-row">
            <div class="sb-market-label">Moneyline</div>
            <div class="sb-market-chips">
              <span class="sb-market-chip home">Home ${h}</span>
              <span class="sb-market-chip draw">Draw ${d}</span>
              <span class="sb-market-chip away">Away ${a}</span>
            </div>
          </div>`;
      }
      // Totals row (pick a representative line, prefer 2.5 if available)
      let tot = '';
      const totals = Array.isArray(mo.totals) ? mo.totals : [];
      if (totals.length) {
        const pick = this.pickTotalsLine(totals);
        const over = this.fmtAmerican(pick?.over?.odds_american);
        const under = this.fmtAmerican(pick?.under?.odds_american);
        const lineTxt = (pick && pick.line != null) ? `${pick.line}` : '';
        tot = `
          <div class="sb-market-row">
            <div class="sb-market-label">Goals O/U ${lineTxt}</div>
            <div class="sb-market-chips">
              <span class="sb-market-chip over">Over ${over}</span>
              <span class="sb-market-chip under">Under ${under}</span>
            </div>
          </div>`;
      }
      // BTTS row
      let btts = '';
      const b = mo.both_teams_to_score || null;
      if (b && (b.yes || b.no)) {
        const yes = this.fmtAmerican(b.yes?.odds_american);
        const no = this.fmtAmerican(b.no?.odds_american);
        btts = `
          <div class="sb-market-row">
            <div class="sb-market-label">BTTS</div>
            <div class="sb-market-chips">
              <span class="sb-market-chip yes">Yes ${yes}</span>
              <span class="sb-market-chip no">No ${no}</span>
            </div>
          </div>`;
      }
      // Corners O/U (pick nearest to 10.5)
      let corners = '';
      const ct = Array.isArray(mo.corners_totals) ? mo.corners_totals : [];
      if (ct.length) {
        const target = 10.5;
        const withLine = ct.filter(t => typeof t.line === 'number');
        const pick = withLine.length ? (withLine.find(t => Math.abs(t.line - target) < 1e-9) || withLine.sort((a,b)=>Math.abs(a.line-target)-Math.abs(b.line-target))[0]) : ct[0];
        const over = this.fmtAmerican(pick?.over?.odds_american);
        const under = this.fmtAmerican(pick?.under?.odds_american);
        const lineTxt = (pick && pick.line != null) ? `${pick.line}` : '';
        corners = `
          <div class="sb-market-row">
            <div class="sb-market-label">Corners O/U ${lineTxt}</div>
            <div class="sb-market-chips">
              <span class="sb-market-chip over">Over ${over}</span>
              <span class="sb-market-chip under">Under ${under}</span>
            </div>
          </div>`;
      }
      if (!ml && !tot && !btts && !corners) return '';
      return `<div class="sb-market-panel">${ml}${tot}${btts}${corners}</div>`;
    } catch { return ''; }
  }

  renderRecommendation(m, oddsRow) {
    try {
      const p = this.getPredRow(m.league, m.game_week, m.home_team, m.away_team);
      if (!p || !p.predictions) return '';
      const pred = p.predictions || {};
      // Basic pick from model result prediction
      const pick = (pred.result_prediction || '').toUpperCase();
      const label = pick === 'H' ? (m.home_team || 'Home') : pick === 'A' ? (m.away_team || 'Away') : 'Draw';
      // Optional: compute simple edge vs moneyline if available
      let edgeTxt = '';
      try {
        const mo = oddsRow && oddsRow.odds && oddsRow.odds.market_odds ? oddsRow.odds.market_odds.match_winner : null;
        if (mo) {
          const ph = typeof pred.home_win_prob === 'number' ? pred.home_win_prob : null;
          const pd = typeof pred.draw_prob === 'number' ? pred.draw_prob : null;
          const pa = typeof pred.away_win_prob === 'number' ? pred.away_win_prob : null;
          const imp = {
            H: this.americanToProb(mo.home?.odds_american),
            D: this.americanToProb(mo.draw?.odds_american),
            A: this.americanToProb(mo.away?.odds_american)
          };
          const mdl = { H: ph, D: pd, A: pa };
          const k = pick;
          if (mdl[k] != null && imp[k] != null && isFinite(mdl[k]) && isFinite(imp[k]) && imp[k] > 0) {
            const edge = (mdl[k] - imp[k]) * 100;
            const sign = edge >= 0 ? '+' : '';
            edgeTxt = ` • Edge ${sign}${edge.toFixed(1)}%`;
          }
        }
      } catch {}
      return `
        <div class="sb-recommendation">
          <span class="sb-chip rec"><i class="fas fa-lightbulb"></i> Pick: ${label}${edgeTxt}</span>
        </div>`;
    } catch { return ''; }
  }

  americanToProb(american) {
    const n = Number(american);
    if (!isFinite(n) || n === 0) return null;
    if (n > 0) return 100 / (n + 100);
    return -n / (-n + 100);
  }

  pickTotalsLine(totals) {
    try {
      // Prefer O/U 2.5; else nearest to 2.5; else first entry
      const target = 2.5;
      const withLine = totals.filter(t => typeof t.line === 'number');
      if (!withLine.length) return totals[0];
      const exact = withLine.find(t => Math.abs(t.line - target) < 1e-9);
      if (exact) return exact;
      let best = withLine[0];
      let bestDiff = Math.abs(best.line - target);
      for (const t of withLine) {
        const d = Math.abs(t.line - target);
        if (d < bestDiff) { best = t; bestDiff = d; }
      }
      return best;
    } catch { return totals[0]; }
  }

  setupNavHandlers() {
    try {
      const sel = document.getElementById('league-select');
      if (sel) {
        sel.addEventListener('change', () => {
          const url = new URL(window.location.href);
          if (sel.value === 'ALL') url.searchParams.delete('league'); else url.searchParams.set('league', sel.value);
          // Keep All Matches to current day when switching leagues (can override via ?today=0)
          url.searchParams.set('today', '1');
          window.location.href = url.toString();
        });
      }
    } catch {}
  }

  // Helpers: local day pruning and date keys
  _localDateKeyFromISO(isoLike) {
    try {
      const d = isoLike ? new Date(isoLike) : null;
      if (!d || isNaN(d.getTime())) return null;
      const y = d.getFullYear();
      const m = String(d.getMonth()+1).padStart(2,'0');
      const da = String(d.getDate()).padStart(2,'0');
      return `${y}-${m}-${da}`;
    } catch { return null; }
  }

  _localDateKeyFromDate(d) {
    try {
      if (!d || isNaN(d.getTime())) return null;
      const y = d.getFullYear();
      const m = String(d.getMonth()+1).padStart(2,'0');
      const da = String(d.getDate()).padStart(2,'0');
      return `${y}-${m}-${da}`;
    } catch { return null; }
  }

  _isoFromLocalKey(key) {
    // Construct an ISO-like string at local midnight to pass to formatter
    // Note: This is only for display in the banner via formatLocalDateParts
    try { return key ? `${key}T00:00:00` : null; } catch { return null; }
  }

  _pruneGroupsToLocalDay(groups, targetLocalKey) {
    try {
      const pruned = [];
      for (const g of (groups || [])) {
        const gKey = targetLocalKey || this._localDateKeyFromISO(g.date);
        if (!gKey) continue;
        const kept = (g.matches || []).filter(m => this._localDateKeyFromISO(m.utc_date || m.date) === gKey);
        if (kept.length) pruned.push({ ...g, date: this._isoFromLocalKey(gKey) || g.date, matches: kept });
      }
      return pruned;
    } catch { return groups || []; }
  }
}

window.AllMatchesManager = AllMatchesManager;
