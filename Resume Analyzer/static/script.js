/* ================================================================
   AI Resume Analyzer — Frontend Logic
   Handles file upload, API calls, and dynamic result rendering
   ================================================================ */

(function () {
    'use strict';

    // ── DOM refs ──────────────────────────────────────────────────
    const dropzone    = document.getElementById('dropzone');
    const fileInput   = document.getElementById('file-input');
    const fileInfo    = document.getElementById('file-info');
    const jobDesc     = document.getElementById('job-desc');
    const analyzeBtn  = document.getElementById('analyze-btn');
    const btnText     = document.getElementById('btn-text');
    const btnSpinner  = document.getElementById('btn-spinner');
    const errorBanner = document.getElementById('error-banner');
    const errorText   = document.getElementById('error-text');
    const results     = document.getElementById('results');

    let selectedFile = null;

    // ── Dimension display config ──────────────────────────────────
    const DIMENSION_META = {
        keyword_match:      { label: 'Keyword Match',      icon: '🎯' },
        section_structure:  { label: 'Section Structure',   icon: '📐' },
        impact_metrics:     { label: 'Impact & Metrics',    icon: '📈' },
        skills_depth:       { label: 'Skills Depth',        icon: '🛠️' },
        formatting:         { label: 'Formatting Quality',  icon: '✍️' },
        action_verbs:       { label: 'Action Verb Usage',   icon: '⚡' },
    };

    // ── Color helpers ─────────────────────────────────────────────
    function scoreColor(score) {
        if (score >= 80) return '#00d2a0';
        if (score >= 60) return '#4fc3f7';
        if (score >= 40) return '#f9a825';
        if (score >= 20) return '#ff8a65';
        return '#ff5252';
    }

    function gradeColor(grade) {
        if (grade.startsWith('A')) return '#00d2a0';
        if (grade.startsWith('B')) return '#4fc3f7';
        if (grade.startsWith('C')) return '#f9a825';
        if (grade.startsWith('D')) return '#ff8a65';
        return '#ff5252';
    }

    // ── Validation ────────────────────────────────────────────────
    function updateBtnState() {
        analyzeBtn.disabled = !(selectedFile && jobDesc.value.trim().length >= 20);
    }

    // ── File handling ─────────────────────────────────────────────
    function handleFile(file) {
        if (!file) return;
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            showError('Please upload a PDF file.');
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            showError('File is too large. Maximum size is 10 MB.');
            return;
        }
        selectedFile = file;
        fileInfo.innerHTML = `✅ ${file.name}`;
        fileInfo.classList.remove('hidden');
        hideError();
        updateBtnState();
    }

    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        handleFile(e.dataTransfer.files[0]);
    });

    jobDesc.addEventListener('input', updateBtnState);

    // ── Error handling ────────────────────────────────────────────
    function showError(msg) {
        errorText.textContent = msg;
        errorBanner.classList.add('visible');
    }
    function hideError() {
        errorBanner.classList.remove('visible');
    }

    // ── Analyze ───────────────────────────────────────────────────
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile || jobDesc.value.trim().length < 20) return;

        hideError();
        results.classList.remove('visible');
        setLoading(true);

        const formData = new FormData();
        formData.append('resume', selectedFile);
        formData.append('job_desc', jobDesc.value.trim());

        try {
            const resp = await fetch('/analyze', {
                method: 'POST',
                body: formData,
            });

            const data = await resp.json();

            if (!resp.ok) {
                showError(data.detail || 'Analysis failed. Please try again.');
                return;
            }

            renderResults(data);
        } catch (err) {
            showError('Network error. Make sure the server is running.');
        } finally {
            setLoading(false);
        }
    });

    function setLoading(on) {
        analyzeBtn.disabled = on;
        btnText.textContent = on ? 'Analyzing...' : '🔍 Analyze Resume';
        btnSpinner.classList.toggle('hidden', !on);
    }

    // ── Render Results ────────────────────────────────────────────
    function renderResults(data) {
        renderScoreHero(data.overall_score, data.grade);
        renderDimensions(data.dimension_scores);
        renderFeedback('suggestions', data.suggestions, 'suggestion');
        renderFeedback('strengths', data.strengths, 'strength');
        renderKeywords(data.keywords_missing);
        renderSections(data.section_analysis);

        results.classList.add('visible');

        // Scroll smoothly to results
        setTimeout(() => {
            results.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    // ── Score Ring Animation ──────────────────────────────────────
    function renderScoreHero(score, grade) {
        const circle = document.getElementById('score-circle');
        const scoreNum = document.getElementById('score-number');
        const gradeBadge = document.getElementById('grade-badge');

        const circumference = 2 * Math.PI * 52; // r=52
        const offset = circumference - (score / 100) * circumference;
        const color = scoreColor(score);

        circle.style.stroke = color;
        circle.style.strokeDasharray = circumference;

        // Trigger animation after a tiny delay
        requestAnimationFrame(() => {
            circle.style.strokeDashoffset = offset;
        });

        // Animate number count up
        animateCount(scoreNum, 0, score, 1200, color);

        // Grade badge
        const gColor = gradeColor(grade);
        gradeBadge.textContent = grade;
        gradeBadge.style.background = gColor + '1a';
        gradeBadge.style.color = gColor;
        gradeBadge.style.border = `2px solid ${gColor}33`;
    }

    function animateCount(el, from, to, duration, color) {
        const start = performance.now();
        el.style.color = color;

        function tick(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = Math.round(from + (to - from) * eased);
            if (progress < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    // ── Dimension Bars ────────────────────────────────────────────
    function renderDimensions(dimScores) {
        const container = document.getElementById('dimensions');
        container.innerHTML = '';

        for (const [key, score] of Object.entries(dimScores)) {
            const meta = DIMENSION_META[key] || { label: key, icon: '📌' };
            const color = scoreColor(score);

            const div = document.createElement('div');
            div.className = 'dimension';
            div.innerHTML = `
                <div class="dimension__header">
                    <span class="dimension__name">${meta.icon} ${meta.label}</span>
                    <span class="dimension__score" style="color: ${color}">${score}</span>
                </div>
                <div class="dimension__bar">
                    <div class="dimension__fill" style="background: linear-gradient(90deg, ${color}cc, ${color})"></div>
                </div>
            `;
            container.appendChild(div);

            // Animate bar fill
            const fill = div.querySelector('.dimension__fill');
            requestAnimationFrame(() => {
                setTimeout(() => {
                    fill.style.width = score + '%';
                }, 100);
            });
        }
    }

    // ── Feedback Lists ────────────────────────────────────────────
    function renderFeedback(containerId, items, type) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';

        if (!items || items.length === 0) {
            const li = document.createElement('li');
            li.className = `feedback-item feedback-item--${type}`;
            li.innerHTML = `<span class="feedback-item__icon">${type === 'strength' ? '✅' : '💡'}</span>
                            <span>${type === 'strength' ? 'No specific strengths identified' : 'No suggestions at this time'}</span>`;
            container.appendChild(li);
            return;
        }

        items.forEach((item, i) => {
            const li = document.createElement('li');
            li.className = `feedback-item feedback-item--${type}`;
            li.style.animationDelay = `${i * 0.08}s`;
            li.innerHTML = `<span class="feedback-item__icon">${type === 'strength' ? '✅' : '💡'}</span>
                            <span>${item}</span>`;
            container.appendChild(li);
        });
    }

    // ── Missing Keywords ──────────────────────────────────────────
    function renderKeywords(keywords) {
        const container = document.getElementById('keywords');
        const card = document.getElementById('keywords-card');
        container.innerHTML = '';

        if (!keywords || keywords.length === 0) {
            card.classList.add('hidden');
            return;
        }

        card.classList.remove('hidden');
        keywords.forEach((kw, i) => {
            const tag = document.createElement('span');
            tag.className = 'keyword-tag';
            tag.textContent = kw;
            tag.style.animationDelay = `${i * 0.06}s`;
            container.appendChild(tag);
        });
    }

    // ── Section Chips ─────────────────────────────────────────────
    function renderSections(sectionAnalysis) {
        const container = document.getElementById('sections');
        container.innerHTML = '';

        if (!sectionAnalysis) return;

        for (const [section, found] of Object.entries(sectionAnalysis)) {
            const chip = document.createElement('div');
            chip.className = `section-chip section-chip--${found ? 'found' : 'missing'}`;
            chip.innerHTML = `<span>${found ? '✅' : '❌'}</span><span>${section}</span>`;
            container.appendChild(chip);
        }
    }

})();
