/**
 * PEDIATRIC BMT SURVIVAL PREDICTOR - CONTROLLER
 * Ultra-simple, fast, interactive controller for real-time survival prognosis.
 */

(function () {
  'use strict';

  // Global State
  const state = {
    activeModel: 'hybrid', // 'hybrid' | 'linear-regression' | 'compare'
    activeCategory: 'all',
    schema: null,
    inputs: {},
    liveCalcEnabled: true,
    debounceTimer: null,
    lastResults: {
      cascade: null,
      lr: null
    }
  };

  // Standard Research Presets
  const PRESETS = {
    median: {
      Recipientage: 9.6,
      Recipientgender: 1,
      Rbodymass: 33.0,
      Recipientage10: 0,
      Recipientageint: 1,
      Donorage: 33.5,
      Donorage35: 0,
      Gendermatch: 0,
      Stemcellsource: 1,
      DonorABO: 0,
      RecipientABO: 0,
      RecipientRh: 1,
      ABOmatch: 1,
      CMVstatus: 2,
      DonorCMV: 0,
      RecipientCMV: 1,
      Disease: 'all',
      Diseasegroup: 1,
      Riskgroup: 0,
      Txpostrelapse: 0,
      Relapse: 0,
      HLAmatch: 0,
      HLAmismatch: 0,
      Antigen: -1,
      Alel: -1,
      HLAgrI: 0,
      CD34kgx10d6: 9.72,
      CD3dCD34: 2.73,
      CD3dkgx10d8: 4.32,
      IIIV: 1,
      aGvHDIIIIV: 1,
      extcGvHD: 1,
      ANCrecovery: 15.0,
      PLTrecovery: 21.0,
      time_to_aGvHD_III_IV: 100.0
    },

    survivor: {
      Recipientage: 8.5,
      Recipientgender: 1,
      Rbodymass: 28.0,
      Recipientage10: 0,
      Recipientageint: 1,
      Donorage: 28.0,
      Donorage35: 0,
      Gendermatch: 1,
      Stemcellsource: 1,
      DonorABO: 0,
      RecipientABO: 0,
      RecipientRh: 1,
      ABOmatch: 1,
      CMVstatus: 0,
      DonorCMV: 0,
      RecipientCMV: 0,
      Disease: 'nonmalignant',
      Diseasegroup: 0,
      Riskgroup: 0,
      Txpostrelapse: 0,
      Relapse: 0,
      HLAmatch: 0,
      HLAmismatch: 0,
      Antigen: -1,
      Alel: -1,
      HLAgrI: 0,
      CD34kgx10d6: 12.5,
      CD3dCD34: 2.5,
      CD3dkgx10d8: 4.8,
      IIIV: 0,
      aGvHDIIIIV: 0,
      extcGvHD: 0,
      ANCrecovery: 13.0,
      PLTrecovery: 18.0,
      time_to_aGvHD_III_IV: 100.0
    },

    highrisk: {
      Recipientage: 16.0,
      Recipientgender: 0,
      Rbodymass: 55.0,
      Recipientage10: 1,
      Recipientageint: 2,
      Donorage: 45.0,
      Donorage35: 1,
      Gendermatch: 0,
      Stemcellsource: 0,
      DonorABO: -1,
      RecipientABO: 1,
      RecipientRh: 0,
      ABOmatch: 0,
      CMVstatus: 3,
      DonorCMV: 1,
      RecipientCMV: 1,
      Disease: 'aml',
      Diseasegroup: 1,
      Riskgroup: 1,
      Txpostrelapse: 1,
      Relapse: 1,
      HLAmatch: 2,
      HLAmismatch: 1,
      Antigen: 2,
      Alel: 2,
      HLAgrI: 3,
      CD34kgx10d6: 3.2,
      CD3dCD34: 8.5,
      CD3dkgx10d8: 2.1,
      IIIV: 1,
      aGvHDIIIIV: 1,
      extcGvHD: 1,
      ANCrecovery: 28.0,
      PLTrecovery: 45.0,
      time_to_aGvHD_III_IV: 25.0
    }
  };

  // Section Mapping Helper
  function normalizeSection(sectionName) {
    if (sectionName === 'Cell Doses & Graft' || sectionName === 'Recovery & GvHD') {
      return 'Graft & Recovery';
    }
    return sectionName || 'Clinical Features';
  }

  // Initialize App on DOM Load
  document.addEventListener('DOMContentLoaded', async () => {
    setupModelSwitcher();
    setupCategoryTabs();
    setupSearchFilter();
    setupPresetsMenu();
    setupHeaderButtons();
    setupFooterActions();

    await loadSchemaAndBuildForm();
    await executePrediction(); // Run initial prediction on load

    if (window.lucide) window.lucide.createIcons();
  });

  // 1. Model Switcher Logic
  function setupModelSwitcher() {
    const buttons = document.querySelectorAll('.btn-model');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        const model = btn.getAttribute('data-model');
        buttons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.activeModel = model;
        updateOutputView();
        executePrediction();
      });
    });
  }

  // Toggle View between Cascade, LR, and Compare
  function updateOutputView() {
    const viewCascade = document.getElementById('view-output-cascade');
    const viewLr = document.getElementById('view-output-lr');
    const viewCompare = document.getElementById('view-output-compare');

    if (viewCascade) viewCascade.style.display = state.activeModel === 'hybrid' ? 'flex' : 'none';
    if (viewLr) viewLr.style.display = state.activeModel === 'linear-regression' ? 'flex' : 'none';
    if (viewCompare) viewCompare.style.display = state.activeModel === 'compare' ? 'flex' : 'none';
  }

  // 2. Category Tab Navigation
  function setupCategoryTabs() {
    const tabs = document.querySelectorAll('.cat-tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        state.activeCategory = tab.getAttribute('data-category');
        applyFeatureFilter();
        const container = document.getElementById('features-container');
        if (container) container.scrollTop = 0;
      });
    });
  }

  // 3. Search and Category Filter Execution
  function setupSearchFilter() {
    const searchInput = document.getElementById('feature-search-input');
    const clearBtn = document.getElementById('btn-clear-search');

    if (searchInput) {
      searchInput.addEventListener('input', () => {
        const val = searchInput.value.trim();
        if (clearBtn) clearBtn.style.display = val ? 'block' : 'none';
        applyFeatureFilter();
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        if (searchInput) searchInput.value = '';
        clearBtn.style.display = 'none';
        applyFeatureFilter();
      });
    }
  }

  function applyFeatureFilter() {
    const searchInput = document.getElementById('feature-search-input');
    const query = (searchInput?.value || '').toLowerCase().trim();
    const items = document.querySelectorAll('.feature-item');
    const groups = document.querySelectorAll('.feature-group-heading');

    let visibleCount = 0;
    const categoryGroupCounts = {};

    items.forEach(item => {
      const featureKey = (item.getAttribute('data-key') || '').toLowerCase();
      const label = (item.getAttribute('data-label') || '').toLowerCase();
      const category = item.getAttribute('data-category') || '';

      const matchesCategory = state.activeCategory === 'all' || category === state.activeCategory;
      const matchesSearch = !query || featureKey.includes(query) || label.includes(query);

      if (matchesCategory && matchesSearch) {
        item.style.display = 'flex';
        visibleCount++;
        categoryGroupCounts[category] = (categoryGroupCounts[category] || 0) + 1;
      } else {
        item.style.display = 'none';
      }
    });

    // Update section headings visibility
    groups.forEach(grp => {
      const cat = grp.getAttribute('data-group-category');
      const count = categoryGroupCounts[cat] || 0;
      grp.style.display = count > 0 ? 'flex' : 'none';
    });

    const badge = document.getElementById('feature-count-badge');
    if (badge) badge.textContent = `${visibleCount} Visible`;
  }

  // 4. Presets Dropdown Menu
  function setupPresetsMenu() {
    const trigger = document.getElementById('btn-presets-menu');
    const menu = document.getElementById('presets-dropdown');

    if (trigger && menu) {
      trigger.addEventListener('click', (e) => {
        e.stopPropagation();
        menu.classList.toggle('show');
      });

      document.addEventListener('click', () => {
        menu.classList.remove('show');
      });

      menu.querySelectorAll('.preset-item').forEach(item => {
        item.addEventListener('click', () => {
          const presetName = item.getAttribute('data-preset');
          applyPreset(presetName);
          menu.classList.remove('show');
        });
      });
    }
  }

  function applyPreset(presetName) {
    const data = PRESETS[presetName] || PRESETS.median;
    Object.keys(data).forEach(key => {
      const val = data[key];
      state.inputs[key] = val;

      const slider = document.getElementById(`slider-${key}`);
      const numInput = document.getElementById(`input-${key}`);
      const select = document.getElementById(`select-${key}`);

      if (slider) slider.value = val;
      if (numInput) numInput.value = val;
      if (select) select.value = val;
    });

    showToast(`Loaded ${presetName.toUpperCase()} preset`);
    executePrediction();
  }

  // 5. Header Action Buttons
  function setupHeaderButtons() {
    // Reset Baseline
    document.getElementById('btn-reset-baseline')?.addEventListener('click', () => {
      applyPreset('median');
    });

    // Run Prediction Button
    document.getElementById('btn-run-prediction')?.addEventListener('click', () => {
      executePrediction();
    });

    // Live Calculation Toggle
    const liveToggle = document.getElementById('toggle-live-calc');
    if (liveToggle) {
      liveToggle.addEventListener('change', () => {
        state.liveCalcEnabled = liveToggle.checked;
      });
    }
  }

  // 6. Schema Loading & Dynamic Control Rendering
  async function loadSchemaAndBuildForm() {
    const container = document.getElementById('features-container');
    try {
      const res = await fetch('/api/schema');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      state.schema = await res.json();
      buildFormControls(state.schema.features);
    } catch (err) {
      console.error('Failed to load schema:', err);
      if (container) {
        container.innerHTML = `
          <div style="padding: 2rem; text-align: center; color: var(--accent-rose);">
            <p>Could not connect to backend server. Make sure API is running.</p>
          </div>
        `;
      }
    }
  }

  function buildFormControls(features) {
    const container = document.getElementById('features-container');
    if (!container) return;
    container.innerHTML = '';

    // Group features by category
    const grouped = {};
    Object.keys(features).forEach(key => {
      const meta = features[key];
      const normSec = normalizeSection(meta.section);
      if (!grouped[normSec]) grouped[normSec] = [];
      grouped[normSec].push({ key, ...meta });

      // Initialize state inputs with defaults
      state.inputs[key] = meta.default !== undefined ? meta.default : (meta.median || 0);
    });

    const orderedCategories = [
      'Demographics',
      'Donor & Matching',
      'Disease & Risk',
      'Histocompatibility (HLA)',
      'Graft & Recovery'
    ];

    orderedCategories.forEach(catName => {
      const list = grouped[catName] || [];
      if (list.length === 0) return;

      // Group Heading
      const heading = document.createElement('div');
      heading.className = 'feature-group-heading';
      heading.setAttribute('data-group-category', catName);
      heading.innerHTML = `<span>${catName}</span> <span style="font-size: 0.7rem; color: var(--text-muted);">${list.length} inputs</span>`;
      container.appendChild(heading);

      // Feature Items
      list.forEach(feat => {
        const item = createFeatureElement(feat, catName);
        container.appendChild(item);
      });
    });

    applyFeatureFilter();
  }

  // Render a Single Feature Element (Slider or Dropdown)
  function createFeatureElement(feat, categoryName) {
    const div = document.createElement('div');
    div.className = 'feature-item';
    div.setAttribute('data-key', feat.key);
    div.setAttribute('data-label', feat.label || feat.key);
    div.setAttribute('data-category', categoryName);

    const isNumeric = feat.type === 'numeric';
    const curVal = state.inputs[feat.key];

    const headerHtml = `
      <div class="feature-header-row">
        <span class="feature-name">${feat.label || feat.key}</span>
        <span class="feature-unit">${feat.unit || ''}</span>
      </div>
    `;

    if (isNumeric) {
      const min = feat.min !== undefined ? feat.min : 0;
      const max = feat.max !== undefined ? feat.max : 100;
      const step = feat.step !== undefined ? feat.step : 0.1;

      div.innerHTML = `
        ${headerHtml}
        <div class="feature-control-row">
          <div class="feature-slider-wrap">
            <input type="range" class="input-range" id="slider-${feat.key}"
              min="${min}" max="${max}" step="${step}" value="${curVal}">
          </div>
          <input type="number" class="input-number" id="input-${feat.key}"
            min="${min}" max="${max}" step="${step}" value="${curVal}">
        </div>
      `;

      const slider = div.querySelector(`#slider-${feat.key}`);
      const numInput = div.querySelector(`#input-${feat.key}`);

      // Two-way synchronization
      slider.addEventListener('input', () => {
        const val = parseFloat(slider.value);
        numInput.value = val;
        state.inputs[feat.key] = val;
        onInputValueChange();
      });

      numInput.addEventListener('input', () => {
        const val = parseFloat(numInput.value);
        if (!isNaN(val)) {
          slider.value = val;
          state.inputs[feat.key] = val;
          onInputValueChange();
        }
      });

    } else {
      // Categorical select dropdown
      const options = feat.options || [];
      const optsHtml = options.map(opt => `
        <option value="${opt.value}" ${opt.value == curVal ? 'selected' : ''}>
          ${opt.label}
        </option>
      `).join('');

      div.innerHTML = `
        ${headerHtml}
        <select class="input-select" id="select-${feat.key}">
          ${optsHtml}
        </select>
      `;

      const select = div.querySelector(`#select-${feat.key}`);
      select.addEventListener('change', () => {
        const val = select.value;
        // Parse int if numeric option
        state.inputs[feat.key] = isNaN(val) ? val : Number(val);
        onInputValueChange();
      });
    }

    return div;
  }

  // Trigger Live Calculation when any control changes
  function onInputValueChange() {
    if (!state.liveCalcEnabled) return;
    clearTimeout(state.debounceTimer);
    state.debounceTimer = setTimeout(() => {
      executePrediction();
    }, 220);
  }

  // 7. Prediction Engine API Execution
  async function executePrediction() {
    const statusText = document.getElementById('output-status-text');
    const runBtn = document.getElementById('btn-run-prediction');
    const runText = document.getElementById('btn-run-text');

    if (statusText) statusText.textContent = 'Calculating...';
    if (runText) runText.textContent = 'Inference...';

    try {
      if (state.activeModel === 'hybrid') {
        const res = await fetch('/api/predict/hybrid', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(state.inputs)
        });
        if (!res.ok) throw new Error(`Cascade API returned ${res.status}`);
        const data = await res.json();
        state.lastResults.cascade = data;
        renderCascadeOutput(data);

      } else if (state.activeModel === 'linear-regression') {
        const res = await fetch('/api/predict/linear-regression', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(state.inputs)
        });
        if (!res.ok) throw new Error(`Linear Regression API returned ${res.status}`);
        const data = await res.json();
        state.lastResults.lr = data;
        renderLrOutput(data);

      } else if (state.activeModel === 'compare') {
        const res = await fetch('/api/predict/compare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(state.inputs)
        });
        if (!res.ok) throw new Error(`Compare API returned ${res.status}`);
        const data = await res.json();
        state.lastResults.cascade = data.model_2_classification;
        state.lastResults.lr = data.model_1_regression;
        renderCompareOutput(data);
      }

      if (statusText) statusText.textContent = 'Live Output';
    } catch (err) {
      console.error('Prediction failed:', err);
      if (statusText) statusText.textContent = 'Offline';
    } finally {
      if (runText) runText.textContent = 'Run Prediction';
      if (window.lucide) window.lucide.createIcons();
    }
  }

  // 8. Render Outputs into Views

  // Cascade Output View
  function renderCascadeOutput(data) {
    const pSurv = data.probability_survived !== undefined ? data.probability_survived : 1.0;
    const pDec = data.probability_deceased !== undefined ? data.probability_deceased : 0.0;
    const isSurvived = data.predicted_class_code === 0;

    // Semicircular Gauge (Arc length ~251.32)
    const gaugeFill = document.getElementById('cascade-gauge-fill');
    const percentEl = document.getElementById('out-cascade-percent');
    const arcCircumference = 251.32;
    const offset = arcCircumference * (1 - pSurv);

    if (gaugeFill) {
      gaugeFill.style.strokeDashoffset = offset;
      gaugeFill.style.stroke = isSurvived ? 'var(--accent-emerald)' : 'var(--accent-rose)';
    }

    if (percentEl) {
      percentEl.textContent = `${(pSurv * 100).toFixed(1)}%`;
      percentEl.style.color = isSurvived ? 'var(--accent-emerald)' : 'var(--accent-rose)';
    }

    // Status Badge
    const statusBadge = document.getElementById('out-cascade-status-badge');
    const statusLabel = document.getElementById('out-cascade-status-label');
    const statusIcon = document.getElementById('out-cascade-status-icon');
    const riskTier = document.getElementById('out-cascade-risk-tier');

    if (statusBadge && statusLabel) {
      statusBadge.className = `outcome-status-badge ${isSurvived ? 'status-survived' : 'status-deceased'}`;
      statusLabel.textContent = isSurvived ? 'HIGH SURVIVAL LIKELIHOOD' : 'ELEVATED MORTALITY RISK';
      if (statusIcon) statusIcon.setAttribute('data-lucide', isSurvived ? 'shield-check' : 'alert-octagon');
    }

    if (riskTier) {
      riskTier.textContent = isSurvived
        ? 'Risk Stratification: Low Risk (Tier 1)'
        : 'Risk Stratification: High Risk (Tier 2)';
    }

    // Metric Values
    document.getElementById('out-cascade-class').textContent = data.predicted_class_label || (isSurvived ? 'Survived (0)' : 'Deceased (1)');
    document.getElementById('out-cascade-p-surv').textContent = `${(pSurv * 100).toFixed(1)}%`;
    document.getElementById('out-cascade-p-dec').textContent = `${(pDec * 100).toFixed(1)}%`;
    document.getElementById('out-cascade-conf').textContent = `${(data.confidence_percentage || 100).toFixed(1)}%`;

    // Clinical Summary
    const interp = document.getElementById('out-cascade-interp');
    if (interp) {
      interp.textContent = data.interpretation || 'Inference successfully executed.';
    }

    // Key factors dynamic text
    const factorHla = document.getElementById('factor-hla');
    const factorCd34 = document.getElementById('factor-cd34');
    const factorAge = document.getElementById('factor-age');

    if (factorHla) {
      const hlaMatch = state.inputs['HLAmatch'] || 0;
      factorHla.textContent = hlaMatch === 0 ? 'Donor-Recipient HLA Match is optimal (10/10).' : `HLA match grade is ${hlaMatch} (mismatch detected).`;
    }
    if (factorCd34) {
      const cd34 = state.inputs['CD34kgx10d6'] || 9.72;
      factorCd34.textContent = cd34 >= 8.0 ? `CD34+ stem cell dose (${cd34} ×10⁶/kg) is in high optimal window.` : `CD34+ dose (${cd34} ×10⁶/kg) is sub-optimal.`;
    }
    if (factorAge) {
      const age = state.inputs['Recipientage'] || 9.6;
      factorAge.textContent = `Pediatric recipient age (${age} years) registered in cohort profile.`;
    }
  }

  // Linear Regression Output View
  function renderLrOutput(data) {
    const days = data.predicted_survival_time_days || 0;
    const years = (days / 365.25).toFixed(1);
    const ci = data.confidence_interval_95_days || { lower: 0, upper: 0 };

    document.getElementById('out-lr-days').textContent = days.toLocaleString();
    document.getElementById('out-lr-years').textContent = `≈ ${years} Years Post-Transplant Observation`;
    document.getElementById('out-lr-ci').textContent = `[${ci.lower?.toLocaleString()}, ${ci.upper?.toLocaleString()}] Days`;

    if (data.model_rmse_days) {
      document.getElementById('out-lr-rmse').textContent = `${data.model_rmse_days.toFixed(1)} d`;
    }
    if (data.model_mae_days) {
      document.getElementById('out-lr-mae').textContent = `${data.model_mae_days.toFixed(1)} d`;
    }

    const interp = document.getElementById('out-lr-interp');
    if (interp) interp.textContent = data.interpretation;
  }

  // Compare Output View
  function renderCompareOutput(data) {
    const c = data.model_2_classification;
    const lr = data.model_1_regression;

    if (c) {
      const isSurv = c.predicted_class_code === 0;
      document.getElementById('cmp-casc-hero').textContent = `${(c.probability_survived * 100).toFixed(1)}%`;
      document.getElementById('cmp-casc-label').textContent = isSurv ? 'High Survival Likelihood' : 'Elevated Mortality Risk';
      document.getElementById('cmp-casc-status').textContent = c.predicted_class_label;
      document.getElementById('cmp-casc-prob').textContent = `${(c.probability_survived * 100).toFixed(1)}%`;
    }

    if (lr) {
      document.getElementById('cmp-lr-hero').innerHTML = `${lr.predicted_survival_time_days.toLocaleString()} <small>d</small>`;
      document.getElementById('cmp-lr-sub').textContent = `≈ ${(lr.predicted_survival_time_days / 365.25).toFixed(1)} Years Post-Transplant`;
      const ci = lr.confidence_interval_95_days;
      document.getElementById('cmp-lr-ci-val').textContent = `[${ci?.lower}, ${ci?.upper}] d`;
    }
  }

  // 9. Copy Summary & Export CSV
  function setupFooterActions() {
    // Copy Summary
    document.getElementById('btn-copy-summary')?.addEventListener('click', () => {
      const summary = generateSummaryText();
      navigator.clipboard.writeText(summary).then(() => {
        showToast('Summary copied to clipboard!');
      }).catch(() => {
        showToast('Clipboard access denied.');
      });
    });

    // Download CSV Record
    document.getElementById('btn-export-csv')?.addEventListener('click', () => {
      downloadPredictionCsv();
    });
  }

  function generateSummaryText() {
    const casc = state.lastResults.cascade;
    const lr = state.lastResults.lr;

    let text = `=== PEDIATRIC BMT SURVIVAL PREDICTOR RECORD ===\n`;
    text += `Generated: ${new Date().toLocaleString()}\n\n`;

    if (casc) {
      text += `[MODEL 2: CASCADE CLASSIFIER]\n`;
      text += `Predicted Status: ${casc.predicted_class_label}\n`;
      text += `Survival Likelihood: ${(casc.probability_survived * 100).toFixed(1)}%\n`;
      text += `Confidence: ${casc.confidence_percentage}%\n\n`;
    }

    if (lr) {
      text += `[MODEL 1: LINEAR REGRESSION]\n`;
      text += `Predicted Survival Time: ${lr.predicted_survival_time_days} Days (~${(lr.predicted_survival_time_days / 365.25).toFixed(1)} Years)\n`;
      text += `95% Confidence Interval: [${lr.confidence_interval_95_days?.lower}, ${lr.confidence_interval_95_days?.upper}] Days\n\n`;
    }

    text += `[KEY PATIENT INPUTS]\n`;
    Object.keys(state.inputs).forEach(k => {
      text += `${k}: ${state.inputs[k]}\n`;
    });

    return text;
  }

  function downloadPredictionCsv() {
    const headers = ['Feature / Metric', 'Value'];
    const rows = [];

    // Model outputs
    if (state.lastResults.cascade) {
      rows.push(['Cascade_Predicted_Status', state.lastResults.cascade.predicted_class_label]);
      rows.push(['Cascade_Survival_Probability', (state.lastResults.cascade.probability_survived * 100).toFixed(1) + '%']);
    }
    if (state.lastResults.lr) {
      rows.push(['Linear_Regression_Predicted_Days', state.lastResults.lr.predicted_survival_time_days]);
    }

    // Features
    Object.keys(state.inputs).forEach(k => {
      rows.push([k, state.inputs[k]]);
    });

    const csvContent = [headers.join(','), ...rows.map(r => `"${r[0]}","${r[1]}"`)].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `bmt_patient_prediction_${Date.now()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    showToast('Downloaded patient record CSV');
  }

  // Toast Notification
  function showToast(message) {
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
      toast.classList.remove('show');
    }, 2800);
  }

})();
