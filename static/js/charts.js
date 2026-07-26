/* ═══════════════════════════════════════════════════════════
   Charts Module — Plotly Dark Theme Charts
   ═══════════════════════════════════════════════════════════ */

const PLOTLY_LAYOUT_BASE = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
        family: 'Plus Jakarta Sans, sans-serif',
        color: '#64748B', /* Slate Gray */
        size: 12,
    },
    margin: { t: 30, r: 20, b: 40, l: 50 },
    showlegend: false,
};

const PLOTLY_CONFIG = {
    displayModeBar: false,
    responsive: true,
};


function renderSegmentPieChart(profiles) {
    const el = document.getElementById('segment-pie-chart');
    if (!el) return;

    const labels = [];
    const values = [];
    const colors = [];

    Object.values(profiles).forEach(p => {
        labels.push(`${p.persona_name}`);
        values.push(p.size);
        colors.push(p.color);
    });

    const data = [{
        type: 'pie',
        labels: labels,
        values: values,
        marker: { colors: colors },
        hole: 0.5,
        textinfo: 'label+percent',
        textposition: 'outside',
        textfont: { size: 11, color: '#64748B' },
        hoverinfo: 'label+value+percent',
        hoverlabel: {
            bgcolor: '#FFFFFF',
            bordercolor: '#C9A66B',
            font: { color: '#2B2B2B', family: 'Plus Jakarta Sans' }
        },
    }];

    const layout = {
        ...PLOTLY_LAYOUT_BASE,
        margin: { t: 10, r: 80, b: 10, l: 80 },
        height: 360,
        annotations: [{
            text: `${Object.keys(profiles).length}<br>Segments`,
            showarrow: false,
            font: { size: 18, color: '#2B2B2B', family: 'Plus Jakarta Sans' },
        }],
    };

    Plotly.newPlot(el, data, layout, PLOTLY_CONFIG);
}


function renderCLVBarChart(profiles) {
    const el = document.getElementById('clv-bar-chart');
    if (!el) return;

    const names = [];
    const clvs = [];
    const colors = [];

    Object.values(profiles).forEach(p => {
        names.push(`${p.persona_name}`);
        clvs.push(p.avg_clv);
        colors.push(p.color);
    });

    const data = [{
        type: 'bar',
        x: names,
        y: clvs,
        marker: {
            color: colors,
            line: { width: 0 },
            opacity: 0.85,
        },
        text: clvs.map(v => '$' + v.toLocaleString()),
        textposition: 'outside',
        textfont: { size: 11, color: '#64748B', family: 'Plus Jakarta Sans' },
        hovertemplate: '%{x}<br>Avg CLV: $%{y:,.0f}<extra></extra>',
        hoverlabel: {
            bgcolor: '#FFFFFF',
            bordercolor: '#C9A66B',
            font: { color: '#2B2B2B' }
        },
    }];

    const layout = {
        ...PLOTLY_LAYOUT_BASE,
        height: 360,
        yaxis: {
            title: 'Avg CLV ($)',
            gridcolor: 'rgba(43,43,43,0.06)',
            zerolinecolor: 'rgba(43,43,43,0.1)',
        },
        xaxis: {
            tickangle: -15,
        },
    };

    Plotly.newPlot(el, data, layout, PLOTLY_CONFIG);
}


function renderSHAPBarChart(shapData) {
    const el = document.getElementById('shap-bar-chart');
    if (!el || !shapData || !shapData.global_importance) return;

    const features = shapData.global_importance.map(d => d.feature).reverse();
    const importances = shapData.global_importance.map(d => d.importance).reverse();
    const directions = shapData.global_importance.map(d => d.direction).reverse();

    const colors = directions.map(d => d === 'positive' ? '#A3B18A' : '#C9A66B');

    const data = [{
        type: 'bar',
        y: features,
        x: importances,
        orientation: 'h',
        marker: {
            color: colors,
            opacity: 0.85,
        },
        text: importances.map(v => v.toFixed(4)),
        textposition: 'outside',
        textfont: { size: 10, color: '#64748B', family: 'Plus Jakarta Sans' },
        hovertemplate: '%{y}<br>Importance: %{x:.4f}<extra></extra>',
        hoverlabel: {
            bgcolor: '#FFFFFF',
            bordercolor: '#C9A66B',
            font: { color: '#2B2B2B' }
        },
    }];

    const layout = {
        ...PLOTLY_LAYOUT_BASE,
        height: 360,
        margin: { t: 10, r: 40, b: 30, l: 120 },
        xaxis: {
            title: 'Mean |SHAP Value|',
            gridcolor: 'rgba(43,43,43,0.06)',
            zerolinecolor: 'rgba(43,43,43,0.1)',
        },
        yaxis: {
            gridcolor: 'rgba(43,43,43,0.06)',
        },
    };

    Plotly.newPlot(el, data, layout, PLOTLY_CONFIG);
}


function renderSilhouetteChart(metricsData) {
    const el = document.getElementById('silhouette-chart');
    if (!el || !metricsData || !metricsData.k_analysis) return;

    const ka = metricsData.k_analysis;

    const data = [{
        type: 'scatter',
        mode: 'lines+markers',
        x: ka.k_range,
        y: ka.silhouettes,
        line: { color: '#4fc3f7', width: 2 },
        marker: {
            color: ka.k_range.map(k => k === ka.optimal_k ? '#ffd700' : '#4fc3f7'),
            size: ka.k_range.map(k => k === ka.optimal_k ? 14 : 8),
            line: { width: 2, color: ka.k_range.map(k => k === ka.optimal_k ? '#ffd700' : 'rgba(0,0,0,0)') },
        },
        text: ka.k_range.map(k => k === ka.optimal_k ? `★ k=${k} (Best)` : `k=${k}`),
        hovertemplate: '%{text}<br>Silhouette: %{y:.4f}<extra></extra>',
        hoverlabel: {
            bgcolor: '#1a1a3e',
            bordercolor: '#4fc3f7',
            font: { color: '#e8e8f0' }
        },
    }];

    const layout = {
        ...PLOTLY_LAYOUT_BASE,
        height: 360,
        xaxis: {
            title: 'Number of Clusters (k)',
            dtick: 1,
            gridcolor: 'rgba(255,255,255,0.04)',
        },
        yaxis: {
            title: 'Silhouette Score',
            gridcolor: 'rgba(255,255,255,0.04)',
        },
        annotations: [{
            x: ka.optimal_k,
            y: ka.silhouettes[ka.k_range.indexOf(ka.optimal_k)],
            text: `Optimal k=${ka.optimal_k}`,
            showarrow: true,
            arrowhead: 2,
            arrowcolor: '#ffd700',
            font: { color: '#ffd700', size: 12, family: 'JetBrains Mono' },
            bgcolor: 'rgba(26, 26, 62, 0.9)',
            bordercolor: '#ffd700',
            borderwidth: 1,
            borderpad: 4,
        }],
    };

    Plotly.newPlot(el, data, layout, PLOTLY_CONFIG);
}


// ── KPI Counter Animation ──────────────────────────────────
function animateCounters() {
    document.querySelectorAll('.kpi-value').forEach(el => {
        const target = parseInt(el.dataset.target);
        if (isNaN(target)) return;

        const prefix = el.dataset.prefix || '';
        const duration = 1500;
        const start = performance.now();

        function update(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(eased * target);
            el.textContent = prefix + current.toLocaleString();

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    });
}
