/* ═══════════════════════════════════════════════════════════
   ROI Calculator — Interactive Revenue Impact Estimation
   ═══════════════════════════════════════════════════════════ */

function initROICalculator(profiles) {
    // Slider value display
    const upliftSlider = document.getElementById('roi-uplift');
    const monthsSlider = document.getElementById('roi-months');
    const upliftDisplay = document.getElementById('uplift-value');
    const monthsDisplay = document.getElementById('months-value');

    if (upliftSlider) {
        upliftSlider.addEventListener('input', () => {
            upliftDisplay.textContent = upliftSlider.value;
        });
    }

    if (monthsSlider) {
        monthsSlider.addEventListener('input', () => {
            monthsDisplay.textContent = monthsSlider.value;
        });
    }

    // Calculate button
    const calcBtn = document.getElementById('calculate-roi-btn');
    if (calcBtn) {
        calcBtn.addEventListener('click', () => calculateROI(profiles));
    }

    // Load all-segments comparison on page load
    loadAllSegmentsROI();
}

async function calculateROI(profiles) {
    const segmentId = document.getElementById('roi-segment').value;
    const uplift = parseInt(document.getElementById('roi-uplift').value);
    const months = parseInt(document.getElementById('roi-months').value);

    try {
        const response = await fetch('/api/roi', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                segment_id: parseInt(segmentId),
                retention_uplift_pct: uplift,
                months: months,
            }),
        });

        const data = await response.json();
        displayROIResults(data);
    } catch (err) {
        console.error('ROI calculation error:', err);
    }
}

function displayROIResults(data) {
    const metricsGrid = document.getElementById('roi-metrics');
    const headline = document.getElementById('roi-headline');
    const chartContainer = document.getElementById('roi-chart-container');

    metricsGrid.style.display = 'grid';
    chartContainer.style.display = 'block';

    headline.innerHTML = `
        <h3 style="color: var(--accent-emerald); font-size: 1.3rem; margin-bottom: 0.5rem;">
            Retaining ${data.customers_saved.toLocaleString()} more customers = 
            <span style="font-family: 'Plus Jakarta Sans'; font-size: 1.5rem; font-weight: 700;">
                $${data.revenue_preserved.toLocaleString()}
            </span> preserved
        </h3>
        <p style="color: var(--text-secondary); font-size: 0.88rem;">
            ${data.retention_uplift_pct}% retention improvement over ${data.projection_months} months for ${data.segment_name}
        </p>
    `;

    document.getElementById('roi-segment-size').textContent = data.segment_size.toLocaleString();
    document.getElementById('roi-at-risk').textContent = '$' + data.revenue_at_risk.toLocaleString();
    document.getElementById('roi-churners-before').textContent = data.expected_churners_without_action.toLocaleString();
    document.getElementById('roi-customers-saved').textContent = data.customers_saved.toLocaleString();
    document.getElementById('roi-revenue-preserved').textContent = '$' + data.revenue_preserved.toLocaleString();
    document.getElementById('roi-multiplier').textContent = data.roi_multiplier + 'x';

    // Comparison chart
    renderROIComparisonChart(data);
}

function renderROIComparisonChart(data) {
    const el = document.getElementById('roi-comparison-chart');
    if (!el) return;

    const plotData = [{
        type: 'bar',
        x: ['Without Campaign', 'With Campaign'],
        y: [data.expected_churners_without_action, data.expected_churners_with_campaign],
        marker: {
            color: ['#C9A66B', '#A3B18A'],
            opacity: 0.85,
        },
        text: [
            data.expected_churners_without_action + ' churners',
            data.expected_churners_with_campaign + ' churners',
        ],
        textposition: 'outside',
        textfont: { size: 12, color: '#64748B', family: 'Plus Jakarta Sans' },
    }];

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Plus Jakarta Sans, sans-serif', color: '#64748B' },
        margin: { t: 30, r: 20, b: 40, l: 50 },
        height: 280,
        yaxis: {
            title: 'Expected Churners',
            gridcolor: 'rgba(43,43,43,0.06)',
        },
        showlegend: false,
    };

    Plotly.newPlot(el, plotData, layout, { displayModeBar: false, responsive: true });
}

async function loadAllSegmentsROI() {
    const el = document.getElementById('all-segments-roi-chart');
    if (!el) return;

    try {
        const response = await fetch('/api/roi/all?uplift=15&months=12');
        const data = await response.json();

        const names = data.map(d => `${d.segment_name}`);
        const atRisk = data.map(d => d.revenue_at_risk);
        const preserved = data.map(d => d.revenue_preserved);

        const plotData = [
            {
                type: 'bar',
                name: 'Revenue at Risk',
                x: names,
                y: atRisk,
                marker: { color: '#C9A66B', opacity: 0.8 },
                text: atRisk.map(v => '$' + v.toLocaleString()),
                textposition: 'outside',
                textfont: { size: 10, color: '#C9A66B', family: 'Plus Jakarta Sans' },
            },
            {
                type: 'bar',
                name: 'Revenue Preserved (15% uplift)',
                x: names,
                y: preserved,
                marker: { color: '#A3B18A', opacity: 0.8 },
                text: preserved.map(v => '$' + v.toLocaleString()),
                textposition: 'outside',
                textfont: { size: 10, color: '#A3B18A', family: 'Plus Jakarta Sans' },
            },
        ];

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { family: 'Plus Jakarta Sans, sans-serif', color: '#64748B' },
            margin: { t: 30, r: 20, b: 60, l: 80 },
            height: 400,
            barmode: 'group',
            yaxis: {
                title: 'Revenue ($)',
                gridcolor: 'rgba(43,43,43,0.06)',
            },
            legend: {
                orientation: 'h',
                y: -0.15,
                font: { size: 11 },
            },
            showlegend: true,
        };

        Plotly.newPlot(el, plotData, layout, { displayModeBar: false, responsive: true });
    } catch (err) {
        console.error('Error loading all segments ROI:', err);
    }
}
