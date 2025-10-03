// Game Week Management for EPL Betting Platform
class GameWeekManager {
    constructor(apiBaseUrl) {
        this.apiBaseUrl = apiBaseUrl;
        this.currentWeek = 1;
        this.gameWeeks = {};
        this.modelPerformance = {};
    this.oddsCompare = null;
    this.currentWeekDetails = null;
    this.totalsCompare = null;
    this.firstHalfCompare = null;
    this.secondHalfCompare = null;
        // Walk-forward summary cache (latest)
        this.walkforward = null;
        // Team totals comparisons (per side)
        this.teamGoalsCompare = { home: null, away: null };
        this.teamCornersCompare = { home: null, away: null };
        this.branding = null; // map of team name -> { primary, secondary, crest }
        this.weekReport = null; // simple aggregate model vs market metrics
    this.bttsCompare = null; // Both Teams To Score comparison (Yes/No)
        // league support
        try {
            const url = new URL(window.location.href);
            this.league = url.searchParams.get('league') || 'PL';
        } catch (e) {
            this.league = 'PL';
        }
        
        this.init();
    }

    // Local time helpers
    formatLocalDateParts(isoLike) {
        try {
            // Prefer explicit UTC date if provided
            const d = isoLike ? new Date(isoLike) : null;
            if (!d || isNaN(d.getTime())) return { date: '—', time: '—', long: '—' };
            const locale = (navigator && navigator.language) ? navigator.language : 'en-US';
            const dateFmt = new Intl.DateTimeFormat(locale, { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
            const timeFmt = new Intl.DateTimeFormat(locale, { hour: 'numeric', minute: '2-digit' });
            const longFmt = new Intl.DateTimeFormat(locale, { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' });
            return {
                date: dateFmt.format(d),
                time: timeFmt.format(d),
                long: longFmt.format(d)
            };
        } catch (e) {
            return { date: '—', time: '—', long: '—' };
        }
    }
    
    // American odds (moneyline) helpers
    probToMoneyline(p) {
        let v = Number(p);
        if (!isFinite(v)) return null;
        if (v <= 0) return null;
        if (v >= 1) return -10000;
        // Clamp to avoid extreme numbers
        v = Math.min(Math.max(v, 0.0001), 0.9999);
        if (v > 0.5) {
            return -Math.round((v / (1 - v)) * 100);
        } else {
            return Math.round(((1 - v) / v) * 100);
        }
    }
    fmtMoneyline(p) {
        const ml = this.probToMoneyline(p);
        if (ml === null || !isFinite(ml)) return '—';
        return (ml > 0 ? '+' : '') + ml;
    }

    // Format American odds integers (e.g., -110, +140). Accepts number or string.
    fmtAmerican(odd) {
        if (odd === null || odd === undefined) return '—';
        const n = Number(odd);
        if (!isFinite(n) || n === 0) return '—';
        return (n > 0 ? '+' : '') + Math.round(n);
    }

    // Convert decimal odds (e.g., 2.50) to American integer (+150)
    decimalToAmerican(dec) {
        const d = Number(dec);
        if (!isFinite(d) || d <= 1) return null;
        if (d >= 2) return Math.round((d - 1) * 100);
        return Math.round(-100 / (d - 1));
    }

    // Result pick -> spelled out label using team names
    formatResultPick(pick, homeTeam, awayTeam) {
        const p = (pick || '').toString().toUpperCase();
        if (p === 'H') return homeTeam || 'Home';
        if (p === 'A') return awayTeam || 'Away';
        if (p === 'D' || p === 'DRAW') return 'Draw';
        return pick || '—';
    }
    
    async init() {
        try {
            await Promise.all([
                this.loadGameWeeks(),
                this.loadBranding()
            ]);
        } catch (e) {
            // Don’t leave the UI stuck on "Loading game weeks..." if the API is slow/unavailable
            console.error('Failed to initialize GameWeekManager:', e);
            // Ensure minimal safe state so render() can show a fallback panel
            this.gameWeeks = this.gameWeeks || {};
            this.weekSummaries = this.weekSummaries || {};
            if (!this.currentWeek || isNaN(this.currentWeek)) this.currentWeek = 1;
        }
        this.render();
        this.setupEventListeners();
    }
    
    async loadGameWeeks() {
        try {
            console.log('🗓️ Loading game weeks...');
            const controller = new AbortController();
            const t = setTimeout(() => controller.abort(), 12000);
            const response = await fetch(`${this.apiBaseUrl}/api/game-weeks?league=${encodeURIComponent(this.league)}`, { signal: controller.signal });
            clearTimeout(t);
            if (!response.ok) throw new Error('Failed to load game weeks');
            
            const data = await response.json();
            this.gameWeeks = data.game_weeks;
            this.weekSummaries = data.week_summaries;
            this.currentWeek = data.current_week;
            // Optional override via query param for deep links
            try {
                const url = new URL(window.location.href);
                const wk = parseInt(url.searchParams.get('week') || '0', 10);
                if (wk && !isNaN(wk) && wk >= 1 && wk <= 38) {
                    this.currentWeek = wk;
                }
            } catch {}
            
            console.log(`✅ Loaded ${data.total_weeks} game weeks`);
        } catch (error) {
            console.error('❌ Error loading game weeks:', error);
            this.showError('Failed to load game weeks: ' + error.message);
        }
    }

    async loadBranding() {
        try {
            // Fetch branding for the active league so crests/logos resolve correctly across leagues
            const resp = await fetch(`${this.apiBaseUrl}/api/branding/teams?league=${encodeURIComponent(this.league)}`);
            if (!resp.ok) throw new Error('Branding fetch failed');
            const data = await resp.json();
            this.branding = data.branding || {};
        } catch (e) {
            console.warn('Branding not available:', e.message);
            this.branding = {};
        }
    }
    
    async loadWeekDetails(week) {
        try {
            console.log(`📊 Loading details for week ${week}...`);
            const controller = new AbortController();
            const t = setTimeout(() => controller.abort(), 12000);
            const response = await fetch(`${this.apiBaseUrl}/api/game-weeks/${week}?league=${encodeURIComponent(this.league)}`, { signal: controller.signal });
            clearTimeout(t);
            if (!response.ok) throw new Error(`Failed to load week ${week} details`);
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error(`❌ Error loading week ${week}:`, error);
            this.showError(`Failed to load week ${week}: ` + error.message);
            return null;
        }
    }
    
    async loadModelPerformance() {
        try {
            console.log('🤖 Loading model performance...');
            const controller = new AbortController();
            const t = setTimeout(() => controller.abort(), 12000);
            const response = await fetch(`${this.apiBaseUrl}/api/model-performance`, { signal: controller.signal });
            clearTimeout(t);
            if (!response.ok) throw new Error('Failed to load model performance');
            
            const data = await response.json();
            this.modelPerformance = data;
            console.log('✅ Model performance loaded');
            return data;
        } catch (error) {
            console.error('❌ Error loading model performance:', error);
            return null;
        }
    }

    async loadWalkforward(rangeHint = 'w1-w5') {
        // Best-effort load of latest walk-forward summary; safe to fail silently
        if (this.walkforward) return this.walkforward;
        try {
            const url = `${this.apiBaseUrl}/api/offline/walkforward/latest${rangeHint ? `?range=${encodeURIComponent(rangeHint)}` : ''}`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`Walk-forward not available (${resp.status})`);
            const data = await resp.json();
            this.walkforward = data.data || null;
            return this.walkforward;
        } catch (e) {
            console.warn('Walk-forward summary unavailable:', e.message || e);
            this.walkforward = null;
            return null;
        }
    }
    
    render() {
        const container = document.getElementById('game-week-container');
        if (!container) return;
        
        container.innerHTML = `
            <div class="game-week-container">
                <div class="week-selector">
                    <div class="week-navigation">
                        <button class="week-nav-btn" id="prev-week" ${this.currentWeek <= 1 ? 'disabled' : ''}>
                            <i class="fas fa-chevron-left"></i> Previous
                        </button>
                        <div class="current-week">
                            Game Week ${this.currentWeek}
                        </div>
                        <button class="week-nav-btn" id="next-week" ${this.currentWeek >= 38 ? 'disabled' : ''}>
                            Next <i class="fas fa-chevron-right"></i>
                        </button>
                        <div class="week-jump">
                            <input type="number" id="jump-week-input" min="1" max="38" placeholder="#" value="${this.currentWeek}" />
                            <button class="week-jump-btn" id="jump-week-btn" title="Jump to week">Go</button>
                            <button class="week-jump-btn" id="save-odds-snapshot" title="Save odds snapshot for this week">Save Odds Snapshot</button>
                        </div>
                    </div>
                    <div class="week-stats" id="week-stats">
                        Loading week stats...
                    </div>
                </div>
                
                <div class="model-performance-summary" id="performance-summary">
                    <div class="performance-header">
                        <h3 class="performance-title">Model Performance</h3>
                        <p class="performance-subtitle">Real-time accuracy tracking and reconciliation</p>
                    </div>
                    <div class="performance-metrics" id="performance-metrics">
                        Loading performance data...
                    </div>
                </div>
                
                <div class="game-cards-grid" id="game-cards">
                    Loading game cards...
                </div>
                <div class="cron-footer" id="cron-footer" style="margin-top:16px;color:#6b7280;font-size:12px;">
                    <span>Sync status: <span id="cron-status-text">loading…</span></span>
                    <button id="cron-retry" class="week-jump-btn" style="margin-left:8px;">Retry hydrate</button>
                </div>
            </div>
        `;
        
        this.renderWeekStats();
        this.renderGameCards();
        this.renderModelPerformance();
        this.attachWeekNavHandlers();
        this.attachCronHelpers();
    }

    async attachCronHelpers() {
        const el = document.getElementById('cron-status-text');
        const btn = document.getElementById('cron-retry');
        if (!el || !btn) return;
        // Load last cron timestamps
        try {
            const r = await fetch(`${this.apiBaseUrl}/api/admin/status/cron-summary`);
            if (r.ok) {
                const data = await r.json();
                // Backend returns a flat object with job names as keys
                const candidates = ['refresh-bovada','snapshot-csv','daily-update','capture-closing'];
                let bestTs = null;
                for (const k of candidates) {
                    const v = data && data[k];
                    const ts = v && v.timestamp ? v.timestamp : null;
                    if (ts) {
                        if (!bestTs || new Date(ts) > new Date(bestTs)) bestTs = ts;
                    }
                }
                if (bestTs) {
                    el.textContent = `last refresh ${new Date(bestTs).toLocaleString()}`;
                } else {
                    el.textContent = 'no recent sync info';
                }
            } else {
                el.textContent = 'status unavailable';
            }
        } catch { el.textContent = 'status unavailable'; }
        // Retry button triggers quick hydrate (refresh -> snapshot CSVs -> precompute -> clear cache -> reload odds)
        btn.onclick = async () => {
            btn.disabled = true; btn.textContent = 'Hydrating…';
            try {
                // 1) Refresh provider snapshots (Bovada EU-first)
                await fetch(`${this.apiBaseUrl}/api/admin/bovada/refresh`, { method: 'POST' });
                // 2) Quick-write CSVs for extended markets to ensure CSV-first reads are populated
                try { await fetch(`${this.apiBaseUrl}/api/admin/odds/snapshot-csv/quick`, { method: 'POST' }); } catch {}
                // 3) Precompute week recommendations (requires dev token in local)
                try { await fetch(`${this.apiBaseUrl}/api/cron/precompute-recommendations?token=dev`, { method: 'POST' }); } catch {}
                // 4) Clear server-side week odds TTL cache, then reload odds
                await fetch(`${this.apiBaseUrl}/api/admin/week-odds-cache/clear`, { method: 'POST' });
                await this.loadWeekOdds(this.currentWeek);
                this.patchCardsWithBookOdds();
                // Refresh cron summary banner after actions
                try {
                    const r = await fetch(`${this.apiBaseUrl}/api/admin/status/cron-summary`);
                    if (r.ok) {
                        const data = await r.json();
                        const stamps = [
                            data && data['refresh-bovada'] && data['refresh-bovada'].timestamp,
                            data && data['snapshot-csv'] && data['snapshot-csv'].timestamp,
                            data && data['precompute-recommendations'] && data['precompute-recommendations'].timestamp,
                        ].filter(Boolean);
                        if (stamps.length) {
                            const latest = stamps.sort().slice(-1)[0];
                            el.textContent = `hydrated at ${new Date(latest).toLocaleString()}`;
                        } else {
                            el.textContent = 'hydrated just now';
                        }
                    } else {
                        el.textContent = 'hydrated just now';
                    }
                } catch { el.textContent = 'hydrated just now'; }
            } catch (e) {
                el.textContent = 'hydrate failed';
            } finally {
                btn.disabled = false; btn.textContent = 'Retry hydrate';
            }
        };
    }

    attachWeekNavHandlers() {
        const prev = document.getElementById('prev-week');
        const next = document.getElementById('next-week');
        const jumpBtn = document.getElementById('jump-week-btn');
        const snapBtn = document.getElementById('save-odds-snapshot');
        if (prev) prev.onclick = () => {
            if (this.currentWeek > 1) {
                this.currentWeek--; this.render();
            }
        };
        if (next) next.onclick = () => {
            if (this.currentWeek < 38) {
                this.currentWeek++; this.render();
            }
        };
        if (jumpBtn) jumpBtn.onclick = () => {
            const val = parseInt(document.getElementById('jump-week-input').value, 10);
            if (!isNaN(val) && val >=1 && val <=38) {
                this.currentWeek = val; this.render();
            }
        };
        if (snapBtn) snapBtn.onclick = async () => {
            try {
                snapBtn.disabled = true; snapBtn.textContent = 'Saving...';
                const resp = await fetch(`${this.apiBaseUrl}/api/game-weeks/${this.currentWeek}/odds-snapshot?edge_threshold=0.05`, { method: 'POST' });
                const data = await resp.json();
                if (resp.ok && data.success) {
                    alert(`Saved odds snapshot to ${data.file}`);
                } else {
                    alert(`Failed to save snapshot: ${data.detail || 'unknown error'}`);
                }
            } catch (e) {
                alert('Snapshot error: ' + (e.message || e));
            } finally {
                snapBtn.disabled = false; snapBtn.textContent = 'Save Odds Snapshot';
            }
        };
    }

    // Removed week report panel and fetch; keeping JS lean at top banner

    setWeek(week) { if (week>=1 && week<=38) { this.currentWeek = week; this.render(); } }
    
    renderWeekStats() {
        const statsContainer = document.getElementById('week-stats');
        if (!statsContainer) return;
        
        const weekSummary = this.weekSummaries?.[this.currentWeek];
        if (!weekSummary) {
            statsContainer.innerHTML = '<div class="week-stat">No data available</div>';
            return;
        }
        
        statsContainer.innerHTML = `
            <div class="week-stat">
                <span class="week-stat-value">${weekSummary.total_matches}</span>
                <span>Matches</span>
            </div>
            <div class="week-stat">
                <span class="week-stat-value">${weekSummary.completed}</span>
                <span>Completed</span>
            </div>
            <div class="week-stat">
                <span class="week-stat-value">${weekSummary.upcoming}</span>
                <span>Upcoming</span>
            </div>
            <div class="week-stat">
                <span class="week-stat-value">${weekSummary.total_goals}</span>
                <span>Goals</span>
            </div>
            <div class="week-stat">
                <span class="week-stat-value">${weekSummary.avg_goals_per_match?.toFixed(1) || '0.0'}</span>
                <span>Avg Goals</span>
            </div>
        `;
    }
    
    async renderGameCards() {
        const cardsContainer = document.getElementById('game-cards');
        if (!cardsContainer) return;
        
        cardsContainer.innerHTML = '<div class="loading">Loading game cards...</div>';
        
    let weekData = await this.loadWeekDetails(this.currentWeek);
    // Fallbacks: if empty, retry with PL, then with ALL to avoid user-facing blanks on hosted envs
    try {
        if (!weekData || !Array.isArray(weekData.matches) || weekData.matches.length === 0) {
            // First fallback: force PL
            const controller1 = new AbortController();
            const t1 = setTimeout(() => controller1.abort(), 12000);
            const r1 = await fetch(`${this.apiBaseUrl}/api/game-weeks/${this.currentWeek}?league=PL`, { signal: controller1.signal });
            clearTimeout(t1);
            if (r1.ok) {
                const d1 = await r1.json();
                if (d1 && Array.isArray(d1.matches) && d1.matches.length > 0) {
                    weekData = d1;
                    this.league = 'PL';
                }
            }
        }
        if (!weekData || !Array.isArray(weekData.matches) || weekData.matches.length === 0) {
            // Second fallback: ALL (server generally defaults to PL)
            const controller2 = new AbortController();
            const t2 = setTimeout(() => controller2.abort(), 12000);
            const r2 = await fetch(`${this.apiBaseUrl}/api/game-weeks/${this.currentWeek}?league=ALL`, { signal: controller2.signal });
            clearTimeout(t2);
            if (r2.ok) {
                const d2 = await r2.json();
                if (d2 && Array.isArray(d2.matches) && d2.matches.length > 0) {
                    weekData = d2;
                    // keep current league selection unchanged for UI, but use data for rendering
                }
            }
        }
    } catch (fallbackErr) {
        console.warn('Week details fallback failed:', fallbackErr?.message || fallbackErr);
    }
    // Fetch comparisons in parallel (after base week data)
        this.loadOddsComparison(this.currentWeek);
        this.loadCornersComparison(this.currentWeek);
    this.loadTotalsComparison(this.currentWeek);
    this.loadFirstHalfComparison(this.currentWeek);
    this.loadSecondHalfComparison(this.currentWeek);
        this.loadBTTSComparison(this.currentWeek);
        this.loadTeamGoalsComparison(this.currentWeek);
        this.loadTeamCornersComparison(this.currentWeek);
        if (!weekData || !weekData.matches) {
            cardsContainer.innerHTML = '<div class="no-matches">No matches found for this week</div>';
            return;
        }
        
        // Save details for per-week performance rendering
        this.currentWeekDetails = weekData;
        const cards = weekData.matches.map(match => this.createGameCard(match)).join('');
            cardsContainer.innerHTML = cards;
            // Patch any sections if data already loaded
            this.patchCardsWithOdds();
            this.patchCardsWithTotals();
            this.patchCardsWithBTTS();
            this.patchCardsWithFirstHalf();
            this.patchCardsWithSecondHalf();
            this.patchCardsWithCorners();
            this.patchCardsWithTeamGoals();
            this.patchCardsWithTeamCorners();
            // Load and render book odds (American)
            try {
                await this.loadWeekOdds(this.currentWeek);
                this.patchCardsWithBookOdds();
            } catch (e) {
                console.warn('Week odds unavailable:', e.message || e);
            }
            // Toggle pills
        this.setupOddsTogglePills();
        // Re-render performance now that we have currentWeekDetails
        this.renderModelPerformance();
    }
    
    createGameCard(match) {
    const statusClass = match.is_completed ? 'completed' : match.is_live ? 'live' : 'upcoming';
    const statusText = match.is_completed ? 'Final' : match.is_live ? 'Live' : 'Scheduled';
        const locked = !!match.is_week_locked;
        const lockMeta = (this.currentWeekDetails && this.currentWeekDetails.week_summary && this.currentWeekDetails.week_summary.lock_info) || null;
    const lockFinal = (lockMeta && lockMeta.finalized_at) ? this.formatLocalDateParts(lockMeta.finalized_at).long : null;
    const lockTip = locked ? `Predictions locked: using closing snapshot${lockFinal ? ` (finalized ${lockFinal})` : ''}` : '';
        const lockBadge = locked ? `<span class="sb-lock-badge" title="${lockTip}"><i class=\"fas fa-lock\"></i> locked</span>` : '';
        const venue = match.venue || match.stadium || 'TBD';
        const weather = match.weather_conditions || match.weather_condition || 'unknown';
        const weatherIcon = this.weatherIcon(weather);
        const predictions = match.predictions || null;
        const oddsRow = this.findOddsComparison(match);
    const corRow = this.findCornersComparison(match);
        const totRow = this.findTotalsComparison(match);
    const bttsRow = this.findBTTSComparison(match);
        const fhRow = this.findFirstHalfComparison(match);
        const shRow = this.findSecondHalfComparison(match);
    const reconciliation = match.reconciliation;
    // Pull correctness from reconciliation (if available)
    const resCorrect = reconciliation ? !!(reconciliation.accuracy && reconciliation.accuracy.result_correct) : null;
    const gpo = reconciliation?.ou?.goals_pred_over;
    const gao = reconciliation?.ou?.goals_actual_over;
    const goalsCorrect = (gpo !== undefined && gpo !== null && gao !== undefined && gao !== null) ? (gpo === gao) : null;
    const cpo = reconciliation?.ou?.corners_pred_over;
    const cao = reconciliation?.ou?.corners_actual_over;
    const cornersCorrect = (cpo !== undefined && cpo !== null && cao !== undefined && cao !== null) ? (cpo === cao) : null;

    const oddsSection = oddsRow ? this.createOddsComparisonSection(oddsRow, { resultCorrect: resCorrect }) : '';
    const cornersSection = corRow ? this.createCornersComparisonSection(corRow, { ouCorrect: cornersCorrect }) : '';
    const totalsSection = totRow ? this.createTotalsComparisonSection(totRow, { ouCorrect: goalsCorrect }) : '';
        const firstHalfSection = fhRow ? this.createFirstHalfComparisonSection(fhRow) : '';
        const secondHalfSection = shRow ? this.createSecondHalfComparisonSection(shRow) : '';
    const edgeBadge = (oddsRow && oddsRow.edge_recommendation) ? `<span class="sb-edge-badge" title="Model edge"><i class=\"fas fa-bolt\"></i><span class=\"edge-val\">${(typeof oddsRow.edge_for_model_pick === 'number') ? oddsRow.edge_for_model_pick.toFixed(2) : ''}</span></span>` : '';
        const kickoffIso = match.utc_date || match.date || null;
        const localParts = this.formatLocalDateParts(kickoffIso);
                // NHL-style head row
                const headRow = `
                        <div class="row head">
                            <div class="game-date js-local-time" title="${localParts.long}">${localParts.date} ${localParts.time}</div>
                            <div class="venue">${venue}</div>
                            <div class="state">${statusText}</div>
                            <div class="period-pill">${match.is_live ? 'Live' : ''}</div>
                            <div class="time-left">—</div>
                        </div>`;
                // NHL-style matchup row with score blocks
                const awayScore = match.is_completed ? (match.away_score ?? 0) : '—';
                const homeScore = match.is_completed ? (match.home_score ?? 0) : '—';
                const matchupRow = `
                        <div class="row matchup sb-match-row">
                            <div class="team side away">
                                <div class="team-line">
                                    <img alt="" title="${match.away_team}" src="${this.getTeamLogo(match.away_team)}" class="team-logo" />
                                    <div class="name">${match.away_team}</div>
                                </div>
                                <div class="score-block">
                                    <div class="live-score js-live-away">${awayScore}</div>
                                    <div class="sub proj-score">—</div>
                                </div>
                            </div>
                            <div class="vs">${match.is_completed ? '' : 'VS'}</div>
                            <div class="team side">
                                <div class="score-block">
                                    <div class="live-score js-live-home">${homeScore}</div>
                                    <div class="sub proj-score">—</div>
                                </div>
                                <div class="team-line">
                                    <img alt="" title="${match.home_team}" src="${this.getTeamLogo(match.home_team)}" class="team-logo" />
                                    <div class="name">${match.home_team}</div>
                                </div>
                            </div>
                        </div>`;
                // Lines summary row (ML and Total)
                const linesRow = `
                        <div class="row details small">
                            <div class="detail-col">
                                <div>
                                    Lines: ML <span class="ml-label">${match.home_team} / ${match.away_team}</span>
                                    · Total ${totRow && totRow.market_line != null ? Number(totRow.market_line).toFixed(1) : '—'}
                                </div>
                            </div>
                        </div>`;
                // Recommendation (moneyline) if edge
                let recRow = '';
                try {
                    if (oddsRow && oddsRow.edge_recommendation && oddsRow.model_pick) {
                        const mp = oddsRow.model_pick;
                        const label = (mp==='H') ? `${match.home_team} ML` : (mp==='A' ? `${match.away_team} ML` : 'Draw');
                        const ev = (typeof oddsRow.ev_for_model_pick==='number') ? oddsRow.ev_for_model_pick : (typeof oddsRow.edge_for_model_pick==='number' ? oddsRow.edge_for_model_pick : null);
                        const bk = oddsRow.ev_bookmaker || '';
                        let oddsStr = '—';
                        try {
                            const dec = oddsRow.preferred_decimals && oddsRow.preferred_decimals[mp];
                            if (typeof dec==='number') oddsStr = this.fmtAmerican(this.decimalToAmerican ? this.decimalToAmerican(dec) : null);
                        } catch {}
                        if (oddsStr==='—') {
                            try {
                                const prob = oddsRow.market_probs && oddsRow.market_probs[mp];
                                if (typeof prob==='number') oddsStr = this.probToMoneyline(prob);
                            } catch {}
                        }
                        const conf = Math.abs(Number(ev||0))>=0.08? 'high' : (Math.abs(Number(ev||0))>=0.05? 'medium' : 'low');
                        const evTxt = (ev!=null) ? `${ev>0?'+':''}${(ev*100).toFixed(1)}%` : '—';
                        recRow = `
                            <div class="row details">
                                <div class="detail-col">
                                    <div class="rec-pill ${ev>0?'ok':(ev===0?'push':'bad')}">
                                        Recommendation: <strong>${label}</strong>
                                        <span class="rec-conf ${conf}">${conf}</span>
                                        · EV ${evTxt}${oddsStr!=='—'?` · ${oddsStr}`:''}${bk?` @ ${bk}`:''}
                                    </div>
                                </div>
                            </div>`;
                    }
                } catch {}
        // Result classification for completed games (W/L/P) based on reconciliation chips
        let resultClass = '';
        try {
            let w = 0, l = 0, p = 0;
            if (resCorrect!=null) { if (resCorrect) w++; else l++; }
            if (goalsCorrect!=null) { if (goalsCorrect) w++; else l++; }
            if (cornersCorrect!=null) { if (cornersCorrect) w++; else l++; }
            if (statusClass==='completed') {
            if (w>0 && l===0) resultClass = 'final-all-win';
            else if (l>0 && w===0 && p===0) resultClass = 'final-all-loss';
            else if (w>0 && l>0) resultClass = 'final-mixed';
            else resultClass = 'final-neutral';
            }
        } catch {}
        const resultBadge = (statusClass==='completed') ? `<div class="result-badge" title="Model vs Actual summary">${resCorrect===true?'W':(resCorrect===false?'L':'')}</div>` : '';
        return `
            <div class="game-card sb-card ${statusClass} card ${resultClass}" data-game-date="${kickoffIso||''}">
                                ${headRow}
                                ${matchupRow}
                                ${linesRow}
                ${recRow}
                                <div class="sb-meta-bottom">
                                        <span class="sb-chip" title="Venue"><i class="fas fa-location-dot"></i> ${venue}</span>
                                        <span class="sb-chip" title="Weather"><i class="fas ${weatherIcon.icon} ${weatherIcon.class}"></i> ${weatherIcon.label}</span>
                                        <span class="sb-chip sb-market-odds" data-home="${match.home_team}" data-away="${match.away_team}" title="Market odds (American)">Odds: —</span>
                    ${resultBadge}
                                </div>
                                <div class="sb-split">
                                        <div class="sb-col predictions-col">${predictions ? this.createPredictionSection(predictions, match.home_team, match.away_team) : '<div class="no-predictions">Predictions unavailable</div>'}</div>
                                        <div class="sb-col odds-col">
                                                <div class="odds-toggle">
                                                        <button class="odds-toggle-pill" aria-expanded="false" title="Show expanded markets">Show Expanded Markets</button>
                                                </div>
                                                <div class="odds-panels-row">
                                                        ${oddsSection || '<div class="odds-placeholder">Market data pending</div>'}
                                                        ${totalsSection}
                                                        ${bttsRow ? this.createBTTSComparisonSection(bttsRow) : ''}
                                                        ${cornersSection}
                                                </div>
                                                <div class="expanded-markets collapsed">
                                                        ${firstHalfSection}
                                                        ${secondHalfSection}
                                                </div>
                                        </div>
                                </div>
                                ${reconciliation ? this.createReconciliationSection(reconciliation) : ''}
                        </div>`;
    }

    // BTTS (Both Teams To Score)
    async loadBTTSComparison(week) {
        try {
            const url = `${this.apiBaseUrl}/api/game-weeks/${week}/btts-compare?league=${encodeURIComponent(this.league)}`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`Failed BTTS compare: HTTP ${resp.status}`);
            this.bttsCompare = await resp.json();
            this.patchCardsWithBTTS();
        } catch (e) {
            console.warn('BTTS comparison unavailable:', e.message || e);
            this.bttsCompare = null;
        }
    }
    findBTTSComparison(match) {
        if (!this.bttsCompare || !this.bttsCompare.matches) return null;
        const date = (match.utc_date || match.date || '').split('T')[0];
        return this.bttsCompare.matches.find(r => r.home_team === match.home_team && r.away_team === match.away_team && (!r.date || !date || r.date === date));
    }
    createBTTSComparisonSection(row) {
        const my = row.model_yes_prob; const ky = row.market_yes_prob; // probabilities
        const pick = row.model_pick || '—';
        const edge = (typeof row.edge_for_yes === 'number') ? row.edge_for_yes.toFixed(2) : '—';
        const mlYes = (row.market_yes_ml != null) ? this.fmtAmerican(row.market_yes_ml) : this.fmtMoneyline(ky);
        const mlNo = (row.market_no_ml != null) ? this.fmtAmerican(row.market_no_ml) : this.fmtMoneyline(ky!=null ? (1-ky) : null);
        const src = row.market_source ? ` <span class="src-badge" title="Market data source">${row.market_source.startsWith('live') ? 'live' : (row.market_source === 'fallback_mock' ? 'mock' : 'historic')}</span>` : '';
        return `
            <div class="odds-compare-section btts ${row.edge_recommendation ? 'mismatch' : ''}">
                <div class="odds-header"><strong>BTTS</strong>${src}</div>
                <div class="odds-probs-grid">
                    <div class="col head">Outcome</div><div class="col head">Model</div><div class="col head">Market</div>
                    <div>Yes</div><div>${my!=null ? (my*100).toFixed(1)+'%' : '—'}</div><div>${ky!=null ? (ky*100).toFixed(1)+'%' : '—'}</div>
                    <div>No</div><div>${my!=null ? ((100-(my*100)).toFixed(1)+'%') : '—'}</div><div>${ky!=null ? ((100-(ky*100)).toFixed(1)+'%') : '—'}</div>
                </div>
                <div class="odds-ml-line">Market: Yes <strong>${mlYes}</strong> • No <strong>${mlNo}</strong></div>
                <div class="edge-line">Model Pick: <strong>${pick}</strong> | Edge: ${edge}</div>
            </div>`;
    }
    patchCardsWithBTTS() {
        if (!this.bttsCompare || !this.bttsCompare.matches) return;
        const container = document.getElementById('game-cards'); if (!container) return;
        const cards = Array.from(container.getElementsByClassName('game-card'));
        cards.forEach(card => {
            try {
                const away = card.querySelector('.sb-team.away .t-name')?.textContent?.trim();
                const home = card.querySelector('.sb-team.home .t-name')?.textContent?.trim();
                if (!away || !home) return;
                const row = this.bttsCompare.matches.find(r => r.home_team === home && r.away_team === away);
                if (!row) return;
                const wrapper = card.querySelector('.sb-col.odds-col .odds-panels-row'); if (!wrapper) return;
                const html = this.createBTTSComparisonSection(row);
                const existing = wrapper.querySelector('.odds-compare-section.btts');
                if (existing) existing.outerHTML = html; else {
                    // Place after totals if present, else before corners if present, else append
                    const totals = wrapper.querySelector('.odds-compare-section.totals');
                    const corners = wrapper.querySelector('.odds-compare-section.corners');
                    if (totals) totals.insertAdjacentHTML('afterend', html);
                    else if (corners) corners.insertAdjacentHTML('beforebegin', html);
                    else wrapper.insertAdjacentHTML('beforeend', html);
                }
                const ph = wrapper.querySelector('.odds-placeholder'); if (ph) ph.remove();
            } catch (e) { console.warn('BTTS patch error:', e); }
        });
    }

    // Week odds: fetch and map by home/away name
    async loadWeekOdds(week) {
    // Prefer CSV/cache-only first with a timeout; optionally try live if needed
        const attempt = async (cacheOnly, timeoutMs) => {
            const controller = new AbortController();
            const t = setTimeout(() => controller.abort(), timeoutMs);
            try {
                const url = `${this.apiBaseUrl}/api/betting/odds/week/${week}?limit=10&league=${encodeURIComponent(this.league)}&cache_only=${cacheOnly ? 'true' : 'false'}`;
                const resp = await fetch(url, { signal: controller.signal });
                clearTimeout(t);
                if (!resp.ok) throw new Error(`Week odds fetch failed (${resp.status})`);
                const data = await resp.json();
                return Array.isArray(data.matches) ? data.matches : [];
            } catch (e) {
                clearTimeout(t);
                throw e;
            }
        };

        try {
            // 1) Cache-only fast path within 6s
            this.weekOdds = await attempt(true, 6000);
        } catch (e1) {
            // 2) Optional live attempt within 8s
            try {
                this.weekOdds = await attempt(false, 8000);
            } catch (e2) {
                console.warn('Week odds unavailable (cache and live failed):', e2?.message || e2);
                this.weekOdds = [];
            }
        }

        // 3) If still empty on first load, try a one-time hydrate: refresh Bovada snapshots, write CSVs, clear week cache, then retry cache-only
        if ((!this.weekOdds || this.weekOdds.length === 0) && !this._hydratedWeekOddsOnce) {
            this._hydratedWeekOddsOnce = true;
            try {
                // Fire-and-forget admin refresh (no auth required) and persist to CSV for durability
                await fetch(`${this.apiBaseUrl}/api/admin/bovada/refresh`, { method: 'POST' });
                await fetch(`${this.apiBaseUrl}/api/admin/odds/snapshot-csv/quick`, { method: 'POST' });
                // Clear server-side week odds TTL cache
                await fetch(`${this.apiBaseUrl}/api/admin/week-odds-cache/clear`, { method: 'POST' });
                // Retry cache-only quickly
                this.weekOdds = await attempt(true, 8000);
            } catch (e3) {
                console.warn('Week odds hydrate retry failed:', e3?.message || e3);
            }
        }

        return this.weekOdds;
    }

    // Extend cron helpers with a quick CSV write action
    async attachCronHelpers() {
        const el = document.getElementById('cron-status-text');
        const btn = document.getElementById('cron-retry');
        if (!el || !btn) return;
        // Load last cron timestamps
        try {
            const r = await fetch(`${this.apiBaseUrl}/api/admin/status/cron-summary`);
            if (r.ok) {
                const data = await r.json();
                const candidates = ['refresh-bovada','snapshot-csv','daily-update','capture-closing'];
                let bestTs = null;
                for (const k of candidates) {
                    const v = data && data[k];
                    const ts = v && v.timestamp ? v.timestamp : null;
                    if (ts) {
                        if (!bestTs || new Date(ts) > new Date(bestTs)) bestTs = ts;
                    }
                }
                if (bestTs) {
                    el.textContent = `last refresh ${new Date(bestTs).toLocaleString()}`;
                } else {
                    el.textContent = 'no recent sync info';
                }
            } else {
                el.textContent = 'status unavailable';
            }
        } catch { el.textContent = 'status unavailable'; }
        // Retry button triggers quick hydrate (refresh -> CSV snapshot -> clear cache -> reload odds)
        btn.onclick = async () => {
            btn.disabled = true; btn.textContent = 'Hydrating…';
            try {
                await fetch(`${this.apiBaseUrl}/api/admin/bovada/refresh`, { method: 'POST' });
                await fetch(`${this.apiBaseUrl}/api/admin/odds/snapshot-csv/quick`, { method: 'POST' });
                await fetch(`${this.apiBaseUrl}/api/admin/week-odds-cache/clear`, { method: 'POST' });
                await this.loadWeekOdds(this.currentWeek);
                this.patchCardsWithBookOdds();
                el.textContent = 'hydrated just now (csv)';
            } catch (e) {
                el.textContent = 'hydrate failed';
            } finally {
                btn.disabled = false; btn.textContent = 'Retry hydrate';
            }
        };
        // Add a quick CSV write action
        const footer = document.getElementById('cron-footer');
        if (footer) {
            const csvBtn = document.createElement('button');
            csvBtn.className = 'week-jump-btn';
            csvBtn.style.marginLeft = '8px';
            csvBtn.textContent = 'Write CSVs';
            footer.appendChild(csvBtn);
            csvBtn.onclick = async () => {
                csvBtn.disabled = true; csvBtn.textContent = 'Writing…';
                try {
                    await fetch(`${this.apiBaseUrl}/api/admin/odds/snapshot-csv/quick`, { method: 'POST' });
                    await fetch(`${this.apiBaseUrl}/api/admin/week-odds-cache/clear`, { method: 'POST' });
                    await this.loadWeekOdds(this.currentWeek);
                    this.patchCardsWithBookOdds();
                    el.textContent = 'csv written just now';
                } catch (e) {
                    el.textContent = 'csv write failed';
                } finally {
                    csvBtn.disabled = false; csvBtn.textContent = 'Write CSVs';
                }
            };
        }
    }

    findWeekOddsRow(match) {
        if (!this.weekOdds) return null;
        const home = match.home_team;
        const away = match.away_team;
        return this.weekOdds.find(r => r.home_team === home && r.away_team === away) || null;
    }

    patchCardsWithBookOdds() {
        if (!this.currentWeekDetails || !Array.isArray(this.currentWeekDetails.matches)) return;
        for (const m of this.currentWeekDetails.matches) {
            const el = document.querySelector(`.sb-market-odds[data-home="${CSS.escape(m.home_team)}"][data-away="${CSS.escape(m.away_team)}"]`);
            if (!el) continue;
            const row = this.findWeekOddsRow(m);
            if (!row || !row.odds || !row.odds.market_odds) continue;
            const mw = row.odds.market_odds.match_winner || {};
            const h = this.fmtAmerican(mw.home?.odds_american);
            const d = this.fmtAmerican(mw.draw?.odds_american);
            const a = this.fmtAmerican(mw.away?.odds_american);
            el.textContent = `Odds: H ${h} | D ${d} | A ${a}`;
        }
    }

    // Team Goals Totals (home/away)
    async loadTeamGoalsComparison(week) {
        try {
            const [homeResp, awayResp] = await Promise.all([
                fetch(`${this.apiBaseUrl}/api/game-weeks/${week}/team-goals-compare?side=home&line=1.5&league=${encodeURIComponent(this.league)}`),
                fetch(`${this.apiBaseUrl}/api/game-weeks/${week}/team-goals-compare?side=away&line=1.5&league=${encodeURIComponent(this.league)}`)
            ]);
            if (homeResp.ok) this.teamGoalsCompare.home = await homeResp.json(); else this.teamGoalsCompare.home = null;
            if (awayResp.ok) this.teamGoalsCompare.away = await awayResp.json(); else this.teamGoalsCompare.away = null;
            this.patchCardsWithTeamGoals();
        } catch (e) {
            console.warn('Team goals comparison unavailable:', e.message || e);
            this.teamGoalsCompare = { home: null, away: null };
        }
    }
    findTeamGoalsRow(match, side) {
        const bucket = (this.teamGoalsCompare && this.teamGoalsCompare[side]) ? this.teamGoalsCompare[side] : null;
        if (!bucket || !bucket.matches) return null;
        const date = (match.utc_date || match.date || '').split('T')[0];
        return bucket.matches.find(r => r.home_team === match.home_team && r.away_team === match.away_team && (!r.date || !date || r.date === date));
    }
    createTeamGoalsComparisonSection(homeRow, awayRow) {
        if (!homeRow && !awayRow) return '';
        const fmt = (p) => p!=null ? (p*100).toFixed(1)+'%' : '—';
        const rowBlock = (row, label) => {
            if (!row) return '';
            const mp = row.model_over_prob; const kp = row.market_over_prob; const line = (row.market_line ?? row.line);
            const pick = row.model_pick || '—';
            const edge = (typeof row.edge_for_model_pick === 'number') ? row.edge_for_model_pick.toFixed(2) : '—';
            const rec = row.edge_recommendation ? '<span class="edge-rec yes">EDGE</span>' : '';
            const fallbackBadge = (row.market_line_fallback && row.market_line && row.line)
                ? `<span class="nearest-badge" title="Requested ${row.line}, used ${row.market_line} (Δ=${(Math.abs(row.market_line - row.line)).toFixed(2)})">nearest</span>`
                : '';
            return `
                <div class="tg-side">
                    <div class="tg-label"><strong>${label}</strong> Line: ${line ?? '—'} ${fallbackBadge}</div>
                    <div class="odds-probs-grid mini">
                        <div class="col head">Outcome</div><div class="col head">Model</div><div class="col head">Market</div>
                        <div>Over</div><div>${fmt(mp)}</div><div>${fmt(kp)}</div>
                        <div>Under</div><div>${mp!=null ? (100 - (mp*100)).toFixed(1)+'%' : '—'}</div><div>${kp!=null ? (100 - (kp*100)).toFixed(1)+'%' : '—'}</div>
                    </div>
                    <div class="odds-ml-line">Market: Over <strong>${this.fmtMoneyline(kp)}</strong> • Under <strong>${this.fmtMoneyline(kp!=null ? (1-kp) : null)}</strong></div>
                    <div class="edge-line">Pick: <strong>${pick}</strong> ${line!=null?`@ ${line}`:''} • Edge: ${edge} ${rec}</div>
                </div>`;
        };
        return `
            <div class="odds-compare-section team-goals">
                <div class="odds-header"><strong>Team Goals O/U</strong>${(homeRow && homeRow.market_source) || (awayRow && awayRow.market_source) ? ` <span class="src-badge" title="Market data source">${((homeRow && homeRow.market_source) || (awayRow && awayRow.market_source)).startsWith('live') ? 'live' : (((homeRow && homeRow.market_source) || (awayRow && awayRow.market_source)) === 'fallback_mock' ? 'mock' : 'historic')}</span>` : ''}</div>
                ${rowBlock(homeRow, 'Home')}
                ${rowBlock(awayRow, 'Away')}
            </div>`;
    }
    patchCardsWithTeamGoals() {
    const container = document.getElementById('game-cards'); if (!container) return;
    const cards = Array.from(container.getElementsByClassName('game-card'));
    if (!cards.length) return;
    cards.forEach(card => {
            try {
                const away = card.querySelector('.sb-team.away .t-name')?.textContent?.trim();
                const home = card.querySelector('.sb-team.home .t-name')?.textContent?.trim();
                if (!away || !home) return;
                // Synthesize a match-like object for find helper
                const mockMatch = { home_team: home, away_team: away, utc_date: null, date: null };
                const homeRow = this.teamGoalsCompare?.home?.matches?.find(r => r.home_team === home && r.away_team === away);
                const awayRow = this.teamGoalsCompare?.away?.matches?.find(r => r.home_team === home && r.away_team === away);
                if (!homeRow && !awayRow) return;
                const html = this.createTeamGoalsComparisonSection(homeRow, awayRow);
                const wrapper = card.querySelector('.sb-col.odds-col .expanded-markets'); if (!wrapper) return;
                const existing = wrapper.querySelector('.odds-compare-section.team-goals');
                if (existing) existing.outerHTML = html; else {
                    wrapper.insertAdjacentHTML('beforeend', html);
                }
                // Remove placeholder if present
                const ph = card.querySelector('.sb-col.odds-col .odds-panels-row .odds-placeholder'); if (ph) ph.remove();
            } catch (e) { console.warn('TeamGoals patch error:', e); }
        });
    }

    // Team Corners Totals (home/away)
    async loadTeamCornersComparison(week) {
        try {
            const [homeResp, awayResp] = await Promise.all([
                fetch(`${this.apiBaseUrl}/api/game-weeks/${week}/team-corners-compare?side=home&line=4.5&league=${encodeURIComponent(this.league)}`),
                fetch(`${this.apiBaseUrl}/api/game-weeks/${week}/team-corners-compare?side=away&line=4.5&league=${encodeURIComponent(this.league)}`)
            ]);
            if (homeResp.ok) this.teamCornersCompare.home = await homeResp.json(); else this.teamCornersCompare.home = null;
            if (awayResp.ok) this.teamCornersCompare.away = await awayResp.json(); else this.teamCornersCompare.away = null;
            this.patchCardsWithTeamCorners();
        } catch (e) {
            console.warn('Team corners comparison unavailable:', e.message || e);
            this.teamCornersCompare = { home: null, away: null };
        }
    }
    createTeamCornersComparisonSection(homeRow, awayRow) {
        if (!homeRow && !awayRow) return '';
        const fmt = (p) => p!=null ? (p*100).toFixed(1)+'%' : '—';
        const rowBlock = (row, label) => {
            if (!row) return '';
            const mp = row.model_over_prob; const kp = row.market_over_prob; const line = row.market_line ?? row.line;
            const pick = row.model_pick || '—';
            const edge = (typeof row.edge_for_model_pick === 'number') ? row.edge_for_model_pick.toFixed(2) : '—';
            const rec = row.edge_recommendation ? '<span class="edge-rec yes">EDGE</span>' : '';
            return `
                <div class="tg-side">
                    <div class="tg-label"><strong>${label}</strong> Line: ${line ?? '—'}</div>
                    <div class="odds-probs-grid mini">
                        <div class="col head">Outcome</div><div class="col head">Model</div><div class="col head">Market</div>
                        <div>Over</div><div>${fmt(mp)}</div><div>${fmt(kp)}</div>
                        <div>Under</div><div>${mp!=null ? (100 - (mp*100)).toFixed(1)+'%' : '—'}</div><div>${kp!=null ? (100 - (kp*100)).toFixed(1)+'%' : '—'}</div>
                    </div>
                    <div class="odds-ml-line">Market: Over <strong>${this.fmtMoneyline(kp)}</strong> • Under <strong>${this.fmtMoneyline(kp!=null ? (1-kp) : null)}</strong></div>
                    <div class="edge-line">Pick: <strong>${pick}</strong> ${line!=null?`@ ${line}`:''} • Edge: ${edge} ${rec}</div>
                </div>`;
        };
        return `
            <div class="odds-compare-section team-corners">
                <div class="odds-header"><strong>Team Corners O/U</strong>${(homeRow && homeRow.market_source) || (awayRow && awayRow.market_source) ? ` <span class="src-badge" title="Market data source">${((homeRow && homeRow.market_source) || (awayRow && awayRow.market_source)).startsWith('live') ? 'live' : (((homeRow && homeRow.market_source) || (awayRow && awayRow.market_source)) === 'fallback_mock' ? 'mock' : 'historic')}</span>` : ''}</div>
                ${rowBlock(homeRow, 'Home')}
                ${rowBlock(awayRow, 'Away')}
            </div>`;
    }
    patchCardsWithTeamCorners() {
    const container = document.getElementById('game-cards'); if (!container) return;
    const cards = Array.from(container.getElementsByClassName('game-card'));
    if (!cards.length) return;
    cards.forEach(card => {
            try {
                const away = card.querySelector('.sb-team.away .t-name')?.textContent?.trim();
                const home = card.querySelector('.sb-team.home .t-name')?.textContent?.trim();
                if (!away || !home) return;
                const homeRow = this.teamCornersCompare?.home?.matches?.find(r => r.home_team === home && r.away_team === away);
                const awayRow = this.teamCornersCompare?.away?.matches?.find(r => r.home_team === home && r.away_team === away);
                if (!homeRow && !awayRow) return;
                const html = this.createTeamCornersComparisonSection(homeRow, awayRow);
                const wrapper = card.querySelector('.sb-col.odds-col .expanded-markets'); if (!wrapper) return;
                const existing = wrapper.querySelector('.odds-compare-section.team-corners');
                if (existing) existing.outerHTML = html; else {
                    wrapper.insertAdjacentHTML('beforeend', html);
                }
                // Remove placeholder if present
                const ph = card.querySelector('.sb-col.odds-col .odds-panels-row .odds-placeholder'); if (ph) ph.remove();
            } catch (e) { console.warn('TeamCorners patch error:', e); }
        });
    }

    // Full-game totals
    async loadTotalsComparison(week) {
        try {
            const url = `${this.apiBaseUrl}/api/game-weeks/${week}/totals-compare?line=2.5&league=${encodeURIComponent(this.league)}`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`Failed totals compare: HTTP ${resp.status}`);
            this.totalsCompare = await resp.json();
            this.patchCardsWithTotals();
        } catch (e) {
            console.warn('Totals comparison unavailable:', e.message);
            this.totalsCompare = null;
        }
    }
    findTotalsComparison(match) {
        if (!this.totalsCompare || !this.totalsCompare.matches) return null;
        const date = (match.utc_date || match.date || '').split('T')[0];
        return this.totalsCompare.matches.find(r => r.home_team === match.home_team && r.away_team === match.away_team && (!r.date || !date || r.date === date));
    }
    createTotalsComparisonSection(row, opts = {}) {
        const mp = row.model_over_prob; const kp = row.market_over_prob; const line = (row.market_line ?? row.line);
        const pick = row.model_pick || '—';
    const edge = (typeof row.edge_for_model_pick === 'number') ? row.edge_for_model_pick.toFixed(2) : '—';
        const rec = row.edge_recommendation ? '<span class="edge-rec yes">EDGE</span>' : '';
        const fallbackBadge = (row.market_line_fallback && row.market_line && row.line)
            ? `<span class="nearest-badge" title="Requested ${row.line}, used ${row.market_line} (Δ=${(Math.abs(row.market_line - row.line)).toFixed(2)})">nearest</span>`
            : '';
        const correctClass = (opts.ouCorrect === true) ? 'correct' : (opts.ouCorrect === false ? 'incorrect' : '');
        return `
            <div class="odds-compare-section totals ${row.edge_recommendation ? 'mismatch' : ''} ${correctClass}">
                <div class="odds-header"><strong>Goals O/U</strong><span>Line: ${line ?? '—'}</span> ${fallbackBadge} ${row.market_source ? ` <span class="src-badge" title="Market data source">${row.market_source.startsWith('live') ? 'live' : (row.market_source === 'fallback_mock' ? 'mock' : 'historic')}</span>` : ''}</div>
                <div class="odds-probs-grid">
                    <div class="col head">Outcome</div><div class="col head">Model</div><div class="col head">Market</div>
                    <div>Over</div><div>${mp!=null ? (mp*100).toFixed(1)+'%' : '—'}</div><div>${kp!=null ? (kp*100).toFixed(1)+'%' : '—'}</div>
                    <div>Under</div><div>${mp!=null ? ((100-(mp*100)).toFixed(1)+'%') : '—'}</div><div>${kp!=null ? ((100-(kp*100)).toFixed(1)+'%') : '—'}</div>
                </div>
                <div class="odds-ml-line">Market: Over <strong>${this.fmtMoneyline(kp)}</strong> • Under <strong>${this.fmtMoneyline(kp!=null ? (1-kp) : null)}</strong></div>
                <div class="edge-line">Model Pick: <strong>${pick}</strong> ${line!=null?`@ ${line}`:''} | Edge: ${edge} ${rec}</div>
            </div>`;
    }
    patchCardsWithTotals() {
        if (!this.totalsCompare || !this.totalsCompare.matches) return;
        const container = document.getElementById('game-cards'); if (!container) return;
        const cards = Array.from(container.getElementsByClassName('game-card'));
        cards.forEach(card => {
            try {
                const away = card.querySelector('.sb-team.away .t-name')?.textContent?.trim();
                const home = card.querySelector('.sb-team.home .t-name')?.textContent?.trim();
                if (!away || !home) return;
                const row = this.totalsCompare.matches.find(r => r.home_team === home && r.away_team === away);
                if (!row) return;
                const wrapper = card.querySelector('.sb-col.odds-col .odds-panels-row'); if (!wrapper) return;
                const existing = wrapper.querySelector('.odds-compare-section.totals');
                const recObj = (this.currentWeekDetails?.matches || []).find(m => (m.home_team === row.home_team && m.away_team === row.away_team));
                const gp = recObj?.reconciliation?.ou?.goals_pred_over;
                const ga = recObj?.reconciliation?.ou?.goals_actual_over;
                const ouCorrect = (gp !== undefined && gp !== null && ga !== undefined && ga !== null) ? (gp === ga) : null;
                const html = this.createTotalsComparisonSection(row, { ouCorrect });
                if (existing) {
                    existing.outerHTML = html;
                } else {
                    // Insert before 1H section if it exists; otherwise before 2H; else at end
                    const firstHalf = wrapper.querySelector('.odds-compare-section.firsthalf');
                    const secondHalf = wrapper.querySelector('.odds-compare-section.secondhalf');
                    if (firstHalf) firstHalf.insertAdjacentHTML('beforebegin', html);
                    else if (secondHalf) secondHalf.insertAdjacentHTML('beforebegin', html);
                    else wrapper.insertAdjacentHTML('beforeend', html);
                }
                // Remove placeholder if present
                const ph = wrapper.querySelector('.odds-placeholder'); if (ph) ph.remove();
            } catch (e) { console.warn('Totals patch error:', e); }
        });
    }

    // First Half totals
    async loadFirstHalfComparison(week) {
        try {
            const url = `${this.apiBaseUrl}/api/game-weeks/${week}/firsthalf-compare?line=1.0&league=${encodeURIComponent(this.league)}`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`Failed first half compare: HTTP ${resp.status}`);
            this.firstHalfCompare = await resp.json();
            this.patchCardsWithFirstHalf();
        } catch (e) { console.warn('First half comparison unavailable:', e.message); this.firstHalfCompare = null; }
    }
    findFirstHalfComparison(match) {
        if (!this.firstHalfCompare || !this.firstHalfCompare.matches) return null;
        const date = (match.utc_date || match.date || '').split('T')[0];
        return this.firstHalfCompare.matches.find(r => r.home_team === match.home_team && r.away_team === match.away_team && (!r.date || !date || r.date === date));
    }
    createFirstHalfComparisonSection(row) {
        const mp = row.model_over_prob; const kp = row.market_over_prob; const line = (row.market_line ?? row.line);
        const pick = row.model_pick || '—';
    const edge = (typeof row.edge_for_model_pick === 'number') ? row.edge_for_model_pick.toFixed(2) : '—';
        const rec = row.edge_recommendation ? '<span class="edge-rec yes">EDGE</span>' : '';
        const fallbackBadge = (row.market_line_fallback && row.market_line && row.line)
            ? `<span class=\"nearest-badge\" title=\"Requested ${row.line}, used ${row.market_line} (Δ=${(Math.abs(row.market_line - row.line)).toFixed(2)})\">nearest</span>`
            : '';
        return `
            <div class="odds-compare-section firsthalf ${row.edge_recommendation ? 'mismatch' : ''}">
                <div class="odds-header"><strong>1H Goals O/U</strong><span>Line: ${line ?? '—'}</span> ${fallbackBadge} ${row.market_source ? ` <span class="src-badge" title="Market data source">${row.market_source.startsWith('live') ? 'live' : (row.market_source === 'fallback_mock' ? 'mock' : 'historic')}</span>` : ''}</div>
                <div class="odds-probs-grid">
                    <div class="col head">Outcome</div><div class="col head">Model</div><div class="col head">Market</div>
                    <div>Over</div><div>${mp!=null ? (mp*100).toFixed(1)+'%' : '—'}</div><div>${kp!=null ? (kp*100).toFixed(1)+'%' : '—'}</div>
                    <div>Under</div><div>${mp!=null ? ((100-(mp*100)).toFixed(1)+'%') : '—'}</div><div>${kp!=null ? ((100-(kp*100)).toFixed(1)+'%') : '—'}</div>
                </div>
                <div class="odds-ml-line">Market: Over <strong>${this.fmtMoneyline(kp)}</strong> • Under <strong>${this.fmtMoneyline(kp!=null ? (1-kp) : null)}</strong></div>
                <div class="edge-line">Model Pick: <strong>${pick}</strong> ${line!=null?`@ ${line}`:''} | Edge: ${edge} ${rec}</div>
            </div>`;
    }
    patchCardsWithFirstHalf() {
        if (!this.firstHalfCompare || !this.firstHalfCompare.matches) return;
        const container = document.getElementById('game-cards'); if (!container) return;
        const cards = Array.from(container.getElementsByClassName('game-card'));
        cards.forEach(card => {
            try {
                const away = card.querySelector('.sb-team.away .t-name')?.textContent?.trim();
                const home = card.querySelector('.sb-team.home .t-name')?.textContent?.trim();
                if (!away || !home) return;
                const row = this.firstHalfCompare.matches.find(r => r.home_team === home && r.away_team === away);
                if (!row) return;
                const wrapper = card.querySelector('.sb-col.odds-col .expanded-markets'); if (!wrapper) return;
                const existing = wrapper.querySelector('.odds-compare-section.firsthalf');
                const html = this.createFirstHalfComparisonSection(row);
                if (existing) existing.outerHTML = html; else {
                    wrapper.insertAdjacentHTML('beforeend', html);
                }
                // Remove placeholder if present
                const ph = card.querySelector('.sb-col.odds-col .odds-panels-row .odds-placeholder'); if (ph) ph.remove();
            } catch (e) { console.warn('FirstHalf patch error:', e); }
        });
    }

    // Second Half totals
    async loadSecondHalfComparison(week) {
        try {
            const url = `${this.apiBaseUrl}/api/game-weeks/${week}/secondhalf-compare?line=1.0&league=${encodeURIComponent(this.league)}`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`Failed second half compare: HTTP ${resp.status}`);
            this.secondHalfCompare = await resp.json();
            this.patchCardsWithSecondHalf();
        } catch (e) { console.warn('Second half comparison unavailable:', e.message); this.secondHalfCompare = null; }
    }
    findSecondHalfComparison(match) {
        if (!this.secondHalfCompare || !this.secondHalfCompare.matches) return null;
        const date = (match.utc_date || match.date || '').split('T')[0];
        return this.secondHalfCompare.matches.find(r => r.home_team === match.home_team && r.away_team === match.away_team && (!r.date || !date || r.date === date));
    }
    createSecondHalfComparisonSection(row) {
        const mp = row.model_over_prob; const kp = row.market_over_prob; const line = (row.market_line ?? row.line);
        const pick = row.model_pick || '—';
    const edge = (typeof row.edge_for_model_pick === 'number') ? row.edge_for_model_pick.toFixed(2) : '—';
        const rec = row.edge_recommendation ? '<span class="edge-rec yes">EDGE</span>' : '';
        const fallbackBadge = (row.market_line_fallback && row.market_line && row.line)
            ? `<span class=\"nearest-badge\" title=\"Requested ${row.line}, used ${row.market_line} (Δ=${(Math.abs(row.market_line - row.line)).toFixed(2)})\">nearest</span>`
            : '';
        return `
            <div class="odds-compare-section secondhalf ${row.edge_recommendation ? 'mismatch' : ''}">
                <div class="odds-header"><strong>2H Goals O/U</strong><span>Line: ${line ?? '—'}</span> ${fallbackBadge} ${row.market_source ? ` <span class="src-badge" title="Market data source">${row.market_source.startsWith('live') ? 'live' : (row.market_source === 'fallback_mock' ? 'mock' : 'historic')}</span>` : ''}</div>
                <div class="odds-probs-grid">
                    <div class="col head">Outcome</div><div class="col head">Model</div><div class="col head">Market</div>
                    <div>Over</div><div>${mp!=null ? (mp*100).toFixed(1)+'%' : '—'}</div><div>${kp!=null ? (kp*100).toFixed(1)+'%' : '—'}</div>
                    <div>Under</div><div>${mp!=null ? ((100-(mp*100)).toFixed(1)+'%') : '—'}</div><div>${kp!=null ? ((100-(kp*100)).toFixed(1)+'%') : '—'}</div>
                </div>
                <div class="odds-ml-line">Market: Over <strong>${this.fmtMoneyline(kp)}</strong> • Under <strong>${this.fmtMoneyline(kp!=null ? (1-kp) : null)}</strong></div>
                <div class="edge-line">Model Pick: <strong>${pick}</strong> ${line!=null?`@ ${line}`:''} | Edge: ${edge} ${rec}</div>
            </div>`;
    }
    patchCardsWithSecondHalf() {
        if (!this.secondHalfCompare || !this.secondHalfCompare.matches) return;
        const container = document.getElementById('game-cards'); if (!container) return;
        const cards = Array.from(container.getElementsByClassName('game-card'));
        cards.forEach(card => {
            try {
                const away = card.querySelector('.sb-team.away .t-name')?.textContent?.trim();
                const home = card.querySelector('.sb-team.home .t-name')?.textContent?.trim();
                if (!away || !home) return;
                const row = this.secondHalfCompare.matches.find(r => r.home_team === home && r.away_team === away);
                if (!row) return;
                const wrapper = card.querySelector('.sb-col.odds-col .expanded-markets'); if (!wrapper) return;
                const existing = wrapper.querySelector('.odds-compare-section.secondhalf');
                const html = this.createSecondHalfComparisonSection(row);
                if (existing) existing.outerHTML = html; else {
                    // Always place 2H after 1H when both are present
                    const firstHalf = wrapper.querySelector('.odds-compare-section.firsthalf');
                    if (firstHalf) firstHalf.insertAdjacentHTML('afterend', html);
                    else wrapper.insertAdjacentHTML('beforeend', html);
                }
                // Remove placeholder if present
                const ph = card.querySelector('.sb-col.odds-col .odds-panels-row .odds-placeholder'); if (ph) ph.remove();
            } catch (e) { console.warn('SecondHalf patch error:', e); }
        });
    }

    async loadOddsComparison(week) {
        try {
            const resp = await fetch(`${this.apiBaseUrl}/api/game-weeks/${week}/odds-compare?league=${encodeURIComponent(this.league)}`);
            if (!resp.ok) throw new Error('Failed odds compare');
            this.oddsCompare = await resp.json();
            // After fetch, update existing cards with odds comparison sections
            this.patchCardsWithOdds();
            this.patchWeekStatsOddsNotice();
        } catch (e) {
            console.warn('Odds comparison unavailable:', e.message);
        }
    }

    findOddsComparison(match) {
        if (!this.oddsCompare || !this.oddsCompare.matches) return null;
        const date = (match.utc_date || match.date || '').split('T')[0];
        const ht = (match.home_team || '').toString();
        const at = (match.away_team || '').toString();
        const htL = ht.toLowerCase();
        const atL = at.toLowerCase();
        return this.oddsCompare.matches.find(r => {
            const rht = (r.home_team || '').toString();
            const rat = (r.away_team || '').toString();
            const rhtN = (r.home_team_normalized || rht).toString();
            const ratN = (r.away_team_normalized || rat).toString();
            const okDate = (!r.date || !date || r.date === date);
            return okDate && (
                (rht === ht && rat === at) ||
                (rhtN === ht && ratN === at) ||
                (rht.toLowerCase() === htL && rat.toLowerCase() === atL) ||
                (rhtN.toLowerCase() === htL && ratN.toLowerCase() === atL)
            );
        });
    }

    patchWeekStatsOddsNotice() {
        try {
            const statsEl = document.getElementById('week-stats');
            if (!statsEl) return;
            // Remove existing notice
            const old = statsEl.querySelector('.week-odds-notice');
            if (old) old.remove();
            const agg = this.oddsCompare?.aggregate;
            if (!agg) return;
            const error = agg.live_odds_error;
            const srcCounts = agg.market_source_counts || {};
            const nonUnknownSources = Object.keys(srcCounts).filter(k => k && k !== 'unknown' && srcCounts[k] > 0);
            // If there's an error or zero sources with live/historic market data, show a small note
            if (error || nonUnknownSources.length === 0) {
                const note = document.createElement('div');
                note.className = 'week-odds-notice';
                let msg = '';
                if (error) {
                    msg = `Live odds unavailable: ${error}`;
                    if (error.toLowerCase().includes('odds_api_key')) {
                        msg += ' • Set ODDS_API_KEY and ODDS_API_REGIONS (e.g., uk,eu,us).';
                    }
                } else {
                    msg = 'No live market odds returned for this week.';
                }
                note.textContent = msg;
                statsEl.appendChild(note);
            }
        } catch (e) {
            // non-fatal
        }
    }

    async loadCornersComparison(week) {
        try {
            const url = `${this.apiBaseUrl}/api/game-weeks/${week}/corners-compare?line=9.5&league=${encodeURIComponent(this.league)}`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`Failed corners compare: HTTP ${resp.status}`);
            this.cornersCompare = await resp.json();
            this.patchCardsWithCorners();
        } catch (e) {
            console.warn('Corners comparison unavailable:', e.message);
            this.cornersCompare = null;
        }
    }

    findCornersComparison(match) {
        if (!this.cornersCompare || !this.cornersCompare.matches) return null;
        const date = (match.utc_date || match.date || '').split('T')[0];
        return this.cornersCompare.matches.find(r => r.home_team === match.home_team && r.away_team === match.away_team && (!r.date || !date || r.date === date));
    }

    createCornersComparisonSection(row, opts = {}) {
        const mp = row.model_over_prob; const kp = row.market_over_prob; const line = (row.market_line ?? row.line);
        const pick = row.model_pick || '—';
    const edge = (typeof row.edge_for_model_pick === 'number') ? row.edge_for_model_pick.toFixed(2) : '—';
        const rec = row.edge_recommendation ? '<span class="edge-rec yes">EDGE</span>' : '';
        const fallbackBadge = (row.market_line_fallback && row.market_line && row.line)
            ? `<span class=\"nearest-badge\" title=\"Requested ${row.line}, used ${row.market_line} (Δ=${(Math.abs(row.market_line - row.line)).toFixed(2)})\">nearest</span>`
            : '';
        const correctClass = (opts.ouCorrect === true) ? 'correct' : (opts.ouCorrect === false ? 'incorrect' : '');
            return `
                <div class="odds-compare-section corners ${row.edge_recommendation ? 'mismatch' : ''} ${correctClass}">
                <div class="odds-header"><strong>Corners O/U</strong><span>Line: ${line ?? '—'}</span> ${fallbackBadge} ${row.market_source ? ` <span class="src-badge" title="Market data source">${row.market_source.startsWith('live') ? 'live' : (row.market_source === 'fallback_mock' ? 'mock' : 'historic')}</span>` : ''}</div>
                <div class="odds-probs-grid">
                    <div class="col head">Outcome</div><div class="col head">Model</div><div class="col head">Market</div>
                    <div>Over</div><div>${mp!=null ? (mp*100).toFixed(1)+'%' : '—'}</div><div>${kp!=null ? (kp*100).toFixed(1)+'%' : '—'}</div>
                    <div>Under</div><div>${mp!=null ? ((100-(mp*100)).toFixed(1)+'%') : '—'}</div><div>${kp!=null ? ((100-(kp*100)).toFixed(1)+'%') : '—'}</div>
                </div>
                <div class="odds-ml-line">Market: Over <strong>${this.fmtMoneyline(kp)}</strong> • Under <strong>${this.fmtMoneyline(kp!=null ? (1-kp) : null)}</strong></div>
                <div class="edge-line">Model Pick: <strong>${pick}</strong> ${line!=null?`@ ${line}`:''} • Edge: ${edge} ${rec}</div>
            </div>`;
    }

    patchCardsWithCorners() {
        if (!this.cornersCompare || !this.cornersCompare.matches) return;
        const container = document.getElementById('game-cards');
        if (!container) return;
        const cards = Array.from(container.getElementsByClassName('game-card'));
        cards.forEach(card => {
            try {
                const away = card.querySelector('.sb-team.away .t-name')?.textContent?.trim();
                const home = card.querySelector('.sb-team.home .t-name')?.textContent?.trim();
                if (!away || !home) return;
                const row = this.cornersCompare.matches.find(r => r.home_team === home && r.away_team === away);
                if (!row) return;
                const wrapper = card.querySelector('.sb-col.odds-col .odds-panels-row'); if (!wrapper) return;
                const existing = wrapper.querySelector('.odds-compare-section.corners');
                const recObj = (this.currentWeekDetails?.matches || []).find(m => (m.home_team === row.home_team && m.away_team === row.away_team));
                const cp = recObj?.reconciliation?.ou?.corners_pred_over;
                const ca = recObj?.reconciliation?.ou?.corners_actual_over;
                const ouCorrect = (cp !== undefined && cp !== null && ca !== undefined && ca !== null) ? (cp === ca) : null;
                const html = this.createCornersComparisonSection(row, { ouCorrect });
                if (existing) {
                    existing.outerHTML = html;
                } else {
                    // Insert corners before half-time panels if they exist
                    const firstHalf = wrapper.querySelector('.odds-compare-section.firsthalf');
                    const secondHalf = wrapper.querySelector('.odds-compare-section.secondhalf');
                    if (firstHalf) firstHalf.insertAdjacentHTML('beforebegin', html);
                    else if (secondHalf) secondHalf.insertAdjacentHTML('beforebegin', html);
                    else wrapper.insertAdjacentHTML('beforeend', html);
                }
                // Remove placeholder if present
                const ph = wrapper.querySelector('.odds-placeholder'); if (ph) ph.remove();
            } catch (e) {
                console.warn('Corners patch error:', e);
            }
        });
    }

    createOddsComparisonSection(row, opts = {}) {
        if (!row.market_probs && !row.model_probs) return '';
        const mp = row.model_probs || {};
        const mkt = row.market_probs || {};
        const pick = row.model_pick;
            const edge = (typeof row.edge_for_model_pick === 'number') ? row.edge_for_model_pick.toFixed(2) : '—';
        const rec = row.edge_recommendation ? '<span class="edge-rec yes">EDGE</span>' : '<span class="edge-rec no">—</span>';
        const mismatch = (pick && row.market_pick && pick !== row.market_pick);
        const correctClass = (opts.resultCorrect === true) ? 'correct' : (opts.resultCorrect === false ? 'incorrect' : '');
        const mlH = this.fmtMoneyline(mkt.H);
        const mlD = this.fmtMoneyline(mkt.D);
        const mlA = this.fmtMoneyline(mkt.A);
        const modelPickLabel = this.formatResultPick(row.model_pick, row.home_team, row.away_team);
        const marketPickLabel = this.formatResultPick(row.market_pick, row.home_team, row.away_team);
        const homeLabel = row.home_team || 'Home';
        const awayLabel = row.away_team || 'Away';
        return `
                    <div class="odds-compare-section result-odds ${mismatch ? 'mismatch' : ''} ${correctClass}">
                        <div class="odds-header">
                            <strong>Model vs Market</strong>
                            <span class="overround">OVR: ${row.market_overround != null ? (row.market_overround*100).toFixed(1)+'%' : '—'}</span>
                            ${row.market_source ? `<span class="src-badge" title="Market data source">${row.market_source.startsWith('live') ? 'live' : (row.market_source === 'fallback_mock' ? 'mock' : 'historic')}</span>` : ''}
                        </div>
                        <div class="odds-probs-grid">
                            <div class="col head">Outcome</div><div class="col head">Model</div><div class="col head">Market</div>
                            <div>${homeLabel}</div><div>${mp.H!=null ? (mp.H*100).toFixed(1)+'%' : '—'}</div><div>${mkt.H != null ? (mkt.H*100).toFixed(1)+'%' : '—'}</div>
                            <div>Draw</div><div>${mp.D!=null ? (mp.D*100).toFixed(1)+'%' : '—'}</div><div>${mkt.D != null ? (mkt.D*100).toFixed(1)+'%' : '—'}</div>
                            <div>${awayLabel}</div><div>${mp.A!=null ? (mp.A*100).toFixed(1)+'%' : '—'}</div><div>${mkt.A != null ? (mkt.A*100).toFixed(1)+'%' : '—'}</div>
                        </div>
                        <div class="odds-ml-line">Market: ${homeLabel} <strong>${mlH}</strong> • Draw <strong>${mlD}</strong> • ${awayLabel} <strong>${mlA}</strong></div>
                        <div class="edge-line">Model Pick: <strong>${modelPickLabel || '—'}</strong> vs Market: <strong>${marketPickLabel || '—'}</strong> | Edge: ${edge} ${rec}</div>
                    </div>`;
    }

    patchCardsWithOdds() {
        if (!this.oddsCompare || !this.oddsCompare.matches) return;
        const container = document.getElementById('game-cards');
        if (!container) return;
        // Re-render each card's odds section (replace placeholder in odds column)
    const cards = Array.from(container.getElementsByClassName('game-card'));
        cards.forEach(card => {
            try {
        const awayEl = card.querySelector('.sb-team.away .t-name');
        const homeEl = card.querySelector('.sb-team.home .t-name');
        if (!awayEl || !homeEl) return;
        const mockMatch = { home_team: homeEl.textContent.trim(), away_team: awayEl.textContent.trim(), utc_date: null, date: null };
        const matchRow = this.findOddsComparison(mockMatch);
                if (!matchRow) return;
                const wrapper = card.querySelector('.sb-col.odds-col .odds-panels-row');
                if (!wrapper) return;
                const recObj = (this.currentWeekDetails?.matches || []).find(m => (m.home_team === mockMatch.home_team && m.away_team === mockMatch.away_team));
                const resultCorrect = recObj?.reconciliation?.accuracy?.result_correct;
                const html = this.createOddsComparisonSection(matchRow, { resultCorrect });
                const existing = wrapper.querySelector('.odds-compare-section.result-odds');
                if (existing) existing.outerHTML = html; else {
                    wrapper.insertAdjacentHTML('afterbegin', html);
                }
                // Remove placeholder if present
                const ph = wrapper.querySelector('.odds-placeholder'); if (ph) ph.remove();
                // Update edge badge near status
                const statusEl = card.querySelector('.sb-status');
                if (statusEl) {
                    const existing = statusEl.querySelector('.sb-edge-badge');
                    if (existing) existing.remove();
                    if (matchRow.edge_recommendation) {
                        const badge = document.createElement('span');
                        badge.className = 'sb-edge-badge';
                        badge.title = 'Model edge';
                        const val = (typeof matchRow.edge_for_model_pick === 'number') ? matchRow.edge_for_model_pick.toFixed(2) : '';
                        badge.innerHTML = `<i class="fas fa-bolt"></i><span class="edge-val">${val}</span>`;
                        statusEl.appendChild(badge);
                    }
                }
            } catch (err) {
                console.warn('Odds patch error:', err);
            }
        });
    }

    weatherIcon(weather) {
        const w = (weather || '').toString().toLowerCase();
        if (!w || w === 'unknown') return { icon: 'fa-cloud', class: 'sb-weather cloud', label: 'TBD' };
        if (w.includes('sun') || w.includes('clear')) return { icon: 'fa-sun', class: 'sb-weather sunny', label: 'Sunny' };
        if (w.includes('rain') || w.includes('showers')) return { icon: 'fa-cloud-showers-heavy', class: 'sb-weather rain', label: 'Rain' };
        if (w.includes('snow')) return { icon: 'fa-snowflake', class: 'sb-weather', label: 'Snow' };
        if (w.includes('cloud')) return { icon: 'fa-cloud', class: 'sb-weather cloud', label: 'Cloudy' };
        return { icon: 'fa-cloud', class: 'sb-weather', label: weather };
    }
    
    createPredictionSection(predictions, homeTeam, awayTeam) {
        const hProb = (typeof predictions.home_win_prob === 'number') ? predictions.home_win_prob : (1 - (predictions.away_win_prob || 0) - (predictions.draw_prob || 0));
        const dProb = (typeof predictions.draw_prob === 'number') ? predictions.draw_prob : 0;
        const aProb = (typeof predictions.away_win_prob === 'number') ? predictions.away_win_prob : 0;
        const homeProb = (hProb * 100).toFixed(1);
        const drawProb = (dProb * 100).toFixed(1);
        const awayProb = (aProb * 100).toFixed(1);
        const resultPick = predictions.result_prediction || (parseFloat(homeProb) > parseFloat(awayProb) && parseFloat(homeProb) > parseFloat(drawProb) ? 'H' : (parseFloat(awayProb) > parseFloat(homeProb) && parseFloat(awayProb) > parseFloat(drawProb) ? 'A' : 'D'));
        const resultPickLabel = this.formatResultPick(resultPick, homeTeam, awayTeam);
        const xg = predictions.xg || {};
        const lambdaHome = (typeof xg.lambda_home === 'number') ? xg.lambda_home.toFixed(2) : '—';
        const lambdaAway = (typeof xg.lambda_away === 'number') ? xg.lambda_away.toFixed(2) : '—';
        const hg = (typeof predictions.home_goals === 'number') ? predictions.home_goals.toFixed(1) : (predictions.home_goals ?? '—');
        const ag = (typeof predictions.away_goals === 'number') ? predictions.away_goals.toFixed(1) : (predictions.away_goals ?? '—');
        const tg = (typeof predictions.home_goals === 'number' || typeof predictions.away_goals === 'number') ? (((predictions.home_goals || 0) + (predictions.away_goals || 0)).toFixed(1)) : '—';

        const confPct = (typeof predictions.confidence === 'number') ? (predictions.confidence * 100).toFixed(1) + '%' : '—';
        const homeLabel = homeTeam || 'Home';
        const awayLabel = awayTeam || 'Away';

        return `
            <div class="model-predictions compact minimalist">
                <div class="prediction-header compact">
                    <div class="model-label"><i class="fas fa-robot"></i> Model</div>
                </div>
                <div class="prediction-lines">
                    <div class="line pick">Pick: <strong>${resultPickLabel}</strong> ${confPct}</div>
                    <div class="line sub">xG λH ${lambdaHome} • λA ${lambdaAway}</div>
                    <div class="line probs h">${homeLabel} ${homeProb}%</div>
                    <div class="line probs d">Draw ${drawProb}%</div>
                    <div class="line probs a">${awayLabel} ${awayProb}%</div>
                    <div class="line metrics">${homeLabel} Goals ${hg}</div>
                    <div class="line metrics">${awayLabel} Goals ${ag}</div>
                    <div class="line metrics">Total Goals ${tg}</div>
                </div>
            </div>`;
    }
    
    createReconciliationSection(reconciliation) {
    const accuracy = reconciliation.accuracy || {};
        // Determine WIN/LOSS/PUSH outcome for moneyline
        const outcome = (accuracy.result_outcome || '').toUpperCase();
        let resultStatus;
        if (outcome === 'WIN' || outcome === 'LOSS' || outcome === 'PUSH') {
            resultStatus = outcome;
        } else if (accuracy.result_push === true) {
            resultStatus = 'PUSH';
        } else if (accuracy.result_correct === true) {
            resultStatus = 'WIN';
        } else if (accuracy.result_correct === false) {
            resultStatus = 'LOSS';
        } else {
            resultStatus = null;
        }
        const resultCorrect = (resultStatus === 'WIN');
        const ou = reconciliation.ou || {};
        const goalsPred = ou.goals_pred_over;
        const goalsAct = ou.goals_actual_over;
        const cornersPred = ou.corners_pred_over;
        const cornersAct = ou.corners_actual_over;
        const goalsHas = (goalsPred !== undefined && goalsPred !== null && goalsAct !== undefined && goalsAct !== null);
        const cornersHas = (cornersPred !== undefined && cornersPred !== null && cornersAct !== undefined && cornersAct !== null);
        const goalsOk = goalsHas ? (goalsPred === goalsAct) : null;
        const cornersOk = cornersHas ? (cornersPred === cornersAct) : null;
        const actual = reconciliation.actual || {};
        const totalGoals = (typeof actual.total_goals === 'number') ? actual.total_goals : ((actual.home_goals || 0) + (actual.away_goals || 0));
        const totalCorners = (typeof actual.total_corners === 'number') ? actual.total_corners : ((actual.home_corners || 0) + (actual.away_corners || 0) || null);
        const chip = (label, val) => {
            let cls = 'neutral'; let text = '—';
            if (val === true) { cls = 'correct'; text = 'Correct'; }
            else if (val === false) { cls = 'incorrect'; text = 'Incorrect'; }
            return `<span class="ou-chip ${cls}">${label}: ${text}</span>`;
        };
        const fmt2 = (v, fallback='—') => (typeof v === 'number' && isFinite(v)) ? Number(v).toFixed(2) : fallback;
        const fmt1p = (v, fallback='—') => (typeof v === 'number' && isFinite(v)) ? (v*100).toFixed(1)+'%' : fallback;
        const pill = resultStatus === 'WIN' ? `<span class="result-pill win" title="Moneyline pick was correct">WIN</span>`
            : resultStatus === 'LOSS' ? `<span class="result-pill loss" title="Moneyline pick was incorrect">LOSS</span>`
            : resultStatus === 'PUSH' ? `<span class="result-pill push" title="Moneyline push">PUSH</span>`
            : '';
        return `
            <div class="reconciliation-section">
                <div class="reconciliation-header">
                    <strong>Model Reconciliation</strong>
                    <div class="accuracy-badge ${resultCorrect ? 'accuracy-correct' : 'accuracy-incorrect'}">
                        Result: ${resultStatus ? (resultStatus === 'WIN' ? 'Correct' : (resultStatus === 'LOSS' ? 'Incorrect' : 'Push')) : '—'} ${pill}
                    </div>
                </div>
                <div class="accuracy-details">
                    <span>Goals Diff: ±${fmt2(accuracy.total_goals_diff)}</span>
                    <span>Within 1 Goal: ${accuracy.goals_within_1 ? 'Yes' : 'No'}</span>
                    <span>Confidence: ${fmt1p(reconciliation.confidence)}</span>
                </div>
                <div class="accuracy-details">
                    <span>Actual Goals: ${typeof totalGoals === 'number' ? totalGoals : '—'}</span>
                    <span>Actual Corners: ${typeof totalCorners === 'number' ? totalCorners : '—'}</span>
                </div>
                <div class="ou-chips-row">
                    ${chip('Goals O/U', goalsOk)}
                    ${chip('Corners O/U', cornersOk)}
                </div>
            </div>
        `;
    }
    
    async renderModelPerformance() {
        const metricsContainer = document.getElementById('performance-metrics');
        if (!metricsContainer) return;
        
        const performanceData = await this.loadModelPerformance();
        // Prefer current week only if available
        const weekPerf = this.currentWeekDetails?.model_performance;
        if (weekPerf && !weekPerf.insufficient_data) {
            const overall = weekPerf;
            const fmtPct = (v) => (v == null ? '—' : (v * 100).toFixed(1) + '%');
            metricsContainer.innerHTML = `
                <div class="metric-card">
                    <div class="metric-value">${fmtPct(overall.result_accuracy)}</div>
                    <div class="metric-label">Week ${this.currentWeek}: Result Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${fmtPct(overall.goals_ou_accuracy)}</div>
                    <div class="metric-label">Week ${this.currentWeek}: Goals O/U Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${fmtPct(overall.corners_ou_accuracy)}</div>
                    <div class="metric-label">Week ${this.currentWeek}: Corners O/U Accuracy</div>
                </div>

            `;
            return;
        }
        // Fallback to overall when per-week not available
        if (!performanceData || !performanceData.overall_performance) {
            metricsContainer.innerHTML = '<div class="metric-card">Performance data unavailable</div>';
            return;
        }
        const overall = performanceData.overall_performance;
        const fmtPct = (v) => (v == null ? '—' : (v * 100).toFixed(1) + '%');
        metricsContainer.innerHTML = `
            <div class="metric-card">
                <div class="metric-value">${fmtPct(overall.result_accuracy)}</div>
                <div class="metric-label">Result Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${fmtPct(overall.goals_ou_accuracy)}</div>
                <div class="metric-label">Goals O/U Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${fmtPct(overall.corners_ou_accuracy)}</div>
                <div class="metric-label">Corners O/U Accuracy</div>
            </div>
        `;
    // Walk-forward strip removed from header
    }

    async renderWalkforwardStrip(metricsContainer) {
        // Disabled: walk-forward chip strip removed from header
        return;
    }
    
    setupEventListeners() {
        document.getElementById('prev-week')?.addEventListener('click', () => {
            if (this.currentWeek > 1) {
                this.currentWeek--;
                this.render();
            }
        });
        
        document.getElementById('next-week')?.addEventListener('click', () => {
            if (this.currentWeek < 38) {
                this.currentWeek++;
                this.render();
            }
        });
    }
    
    getTeamLogo(teamName) {
        // Prefer exact matches from branding; fallback to placeholder
        const b = this.branding && this.branding[teamName];
        if (b && b.crest) return b.crest;
        // Handle common short names to full names
        const map = {
            'Man City': 'Manchester City FC',
            'Manchester City': 'Manchester City FC',
            'Man United': 'Manchester United FC',
            'Manchester United': 'Manchester United FC',
            'Tottenham': 'Tottenham Hotspur FC',
            'West Ham': 'West Ham United FC',
            'Newcastle': 'Newcastle United FC',
            'Leeds United': 'Leeds United FC',
            // Common Bundesliga short/canonical variants
            'Bayern': 'FC Bayern München',
            'Leverkusen': 'Bayer 04 Leverkusen',
            'Dortmund': 'Borussia Dortmund',
            'Gladbach': "Borussia Mönchengladbach",
            'Mönchengladbach': "Borussia Mönchengladbach",
            'Monchengladbach': "Borussia Mönchengladbach",
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
            'Bremen': 'SV Werder Bremen'
        };
        const full = map[teamName];
        if (full && this.branding && this.branding[full]?.crest) return this.branding[full].crest;
        // Try case-insensitive lookup
        const tryLower = teamName && this.branding && Object.keys(this.branding).find(n => n.toLowerCase() === teamName.toLowerCase());
        if (tryLower && this.branding[tryLower]?.crest) return this.branding[tryLower].crest;
        return 'https://via.placeholder.com/32x32/cccccc/666666?text=' + (teamName?.charAt(0) || '?');
    }
    
    setupOddsTogglePills() {
        const container = document.getElementById('game-cards');
        if (!container) return;
        const pills = Array.from(container.querySelectorAll('.odds-toggle-pill'));
        pills.forEach(pill => {
            pill.onclick = () => {
                const oddsCol = pill.closest('.sb-col.odds-col');
                if (!oddsCol) return;
                const expanded = oddsCol.querySelector('.expanded-markets');
                if (!expanded) return;
                const isCollapsed = expanded.classList.contains('collapsed');
                if (isCollapsed) {
                    expanded.classList.remove('collapsed');
                    pill.setAttribute('aria-expanded', 'true');
                    pill.textContent = 'Hide Expanded Markets';
                    pill.title = 'Hide expanded markets';
                } else {
                    expanded.classList.add('collapsed');
                    pill.setAttribute('aria-expanded', 'false');
                    pill.textContent = 'Show Expanded Markets';
                    pill.title = 'Show expanded markets';
                }
            };
        });
    }
    
    showError(message) {
        console.error('Error:', message);
        // You can implement toast notifications here
    }
}

// Export for use in main app
window.GameWeekManager = GameWeekManager;