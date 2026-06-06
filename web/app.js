/* ===================================================================
   app.js -- Three.js Drone + Neural Network Monitor + REST API Polling
   =================================================================== */

// -- State ------------------------------------------------------------
let scene, camera, renderer, drone, roaSphere, controls;
let goalMarker, obstacles3D = [];
let simRunning = false;
let lyapHistory = [], barrierHistory = [];
const MAX_HISTORY = 300;
let pollInterval = null;

// -- Initialise Three.js ----------------------------------------------
function initScene() {
    const container = document.getElementById('viewport-container');
    const w = container.clientWidth;
    const h = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060a14);
    scene.fog = new THREE.FogExp2(0x060a14, 0.03);

    camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 500);
    camera.position.set(12, 12, 8);
    camera.lookAt(5, 5, 2);

    renderer = new THREE.WebGLRenderer({
        canvas: document.getElementById('three-canvas'),
        antialias: true
    });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Grid
    const grid = new THREE.GridHelper(20, 20, 0x1a2744, 0x111a30);
    grid.rotation.x = Math.PI / 2;
    grid.position.set(5, 5, 0);
    scene.add(grid);

    // Lights
    scene.add(new THREE.AmbientLight(0x334466, 0.6));
    const dirLight = new THREE.DirectionalLight(0x6699ff, 0.8);
    dirLight.position.set(10, 10, 15);
    scene.add(dirLight);

    // Drone model
    createDrone();

    // ROA sphere
    const roaGeo = new THREE.SphereGeometry(1.5, 32, 32);
    const roaMat = new THREE.MeshBasicMaterial({
        color: 0x3b82f6, transparent: true, opacity: 0.08,
        side: THREE.DoubleSide, depthWrite: false
    });
    roaSphere = new THREE.Mesh(roaGeo, roaMat);
    roaSphere.position.set(5, 5, 2);
    scene.add(roaSphere);

    // Goal marker
    const goalGeo = new THREE.OctahedronGeometry(0.15, 0);
    const goalMat = new THREE.MeshBasicMaterial({ color: 0x22c55e });
    goalMarker = new THREE.Mesh(goalGeo, goalMat);
    goalMarker.position.set(5, 5, 2);
    scene.add(goalMarker);

    // Axes
    scene.add(new THREE.AxesHelper(2));

    // Resize
    window.addEventListener('resize', () => {
        const w2 = container.clientWidth;
        const h2 = container.clientHeight;
        camera.aspect = w2 / h2;
        camera.updateProjectionMatrix();
        renderer.setSize(w2, h2);
    });

    // Click handlers
    container.addEventListener('click', onViewportClick);
    container.addEventListener('contextmenu', onViewportRightClick);

    // OrbitControls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.target.set(5, 5, 2);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 3;
    controls.maxDistance = 50;
    controls.update();
}

function createDrone() {
    drone = new THREE.Group();

    // Body
    const bodyGeo = new THREE.BoxGeometry(0.15, 0.15, 0.06);
    const bodyMat = new THREE.MeshPhongMaterial({ color: 0x1e40af, emissive: 0x0a1628, shininess: 80 });
    drone.add(new THREE.Mesh(bodyGeo, bodyMat));

    // Arms + Propellers
    const armGeo = new THREE.CylinderGeometry(0.008, 0.008, 0.36, 6);
    const armMat = new THREE.MeshPhongMaterial({ color: 0x334155 });
    const propGeo = new THREE.CircleGeometry(0.08, 16);
    const propColors = [0x3b82f6, 0x22c55e, 0xf59e0b, 0xef4444];
    const angles = [Math.PI / 4, -Math.PI / 4, 3 * Math.PI / 4, -3 * Math.PI / 4];

    drone.propellers = [];
    for (let i = 0; i < 4; i++) {
        const arm = new THREE.Mesh(armGeo, armMat);
        arm.rotation.z = angles[i];
        drone.add(arm);

        const px = Math.cos(angles[i]) * 0.18;
        const py = Math.sin(angles[i]) * 0.18;
        const propMat = new THREE.MeshBasicMaterial({
            color: propColors[i], transparent: true, opacity: 0.6, side: THREE.DoubleSide
        });
        const prop = new THREE.Mesh(propGeo, propMat);
        prop.position.set(px, py, 0.04);
        drone.add(prop);
        drone.propellers.push(prop);
    }

    // LED
    const ledMat = new THREE.MeshBasicMaterial({ color: 0x22c55e });
    drone.led = new THREE.Mesh(new THREE.SphereGeometry(0.02, 8, 8), ledMat);
    drone.led.position.set(0, 0, -0.04);
    drone.add(drone.led);

    // Trail
    drone.trail = [];
    drone.trailLine = null;
    drone.position.set(5, 5, 2);
    scene.add(drone);
}

// -- Obstacle 3D creation ---------------------------------------------
function addObstacle3D(cx, cy, cz, radius) {
    const geo = new THREE.SphereGeometry(radius, 20, 20);
    const mat = new THREE.MeshPhongMaterial({
        color: 0xef4444, transparent: true, opacity: 0.35, emissive: 0x330808
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(cx, cy, cz);
    const wire = new THREE.Mesh(
        new THREE.SphereGeometry(radius + 0.01, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0xff3333, wireframe: true, transparent: true, opacity: 0.2 })
    );
    mesh.add(wire);
    scene.add(mesh);
    obstacles3D.push({ mesh, data: { center: [cx, cy, cz], radius } });
}

function clearObstacles3D() {
    for (const obs of obstacles3D) scene.remove(obs.mesh);
    obstacles3D = [];
}

// -- Click Handlers ---------------------------------------------------
function onViewportClick(e) {
    if (e.button !== 0) return;
    const rect = e.target.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -2);
    const pt = new THREE.Vector3();
    raycaster.ray.intersectPlane(plane, pt);
    if (pt) {
        goalMarker.position.copy(pt);
        fetch('/api/goal', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ position: [pt.x, pt.y, pt.z] })
        }).catch(() => { });
    }
}

function onViewportRightClick(e) {
    e.preventDefault();
    const rect = e.target.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -2);
    const pt = new THREE.Vector3();
    raycaster.ray.intersectPlane(plane, pt);
    if (pt) {
        const r = 0.3 + Math.random() * 0.3;
        addObstacle3D(pt.x, pt.y, pt.z, r);
        fetch('/api/obstacle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ center: [pt.x, pt.y, pt.z], radius: r })
        }).catch(() => { });
    }
}

// -- Update Drone Visual ----------------------------------------------
function updateDrone(state, status) {
    if (!state || !drone) return;
    drone.position.set(state[0], state[1], state[2]);
    if (state.length >= 10) {
        drone.quaternion.set(state[7], state[8], state[9], state[6]);
    }

    // Propellers
    if (status && status.motor_rpms) {
        for (let i = 0; i < 4; i++) {
            const rpm = status.motor_rpms[i] || 3980;
            drone.propellers[i].rotation.z += (rpm / 12000) * 0.5;
            const r = Math.min(rpm / 12000, 1);
            drone.propellers[i].material.color.setHSL((1 - r) * 120 / 360, 1, 0.5);
        }
    }

    // LED
    if (status) {
        drone.led.material.color.setHex(status.is_stable ? 0x22c55e : 0xef4444);
    }

    // ROA sphere
    roaSphere.position.copy(drone.position);
    if (status && status.roa_remaining !== undefined) {
        const s = Math.max(0.1, status.roa_remaining * 2);
        roaSphere.scale.set(s, s, s);
    }

    // Goal marker
    if (status && status.goal) {
        goalMarker.position.set(status.goal[0], status.goal[1], status.goal[2]);
    }

    // Trail
    drone.trail.push(drone.position.clone());
    if (drone.trail.length > 500) drone.trail.shift();
    if (drone.trailLine) scene.remove(drone.trailLine);
    if (drone.trail.length > 2) {
        const geo = new THREE.BufferGeometry().setFromPoints(drone.trail);
        drone.trailLine = new THREE.Line(geo,
            new THREE.LineBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.3 }));
        scene.add(drone.trailLine);
    }
}

// -- Update Neural Network Monitor ------------------------------------
function updateNNPanel(nn) {
    if (!nn) return;

    // Decision
    const dec = nn.decision || {};
    const decEl = document.getElementById('nn-decision');
    const reasonEl = document.getElementById('nn-reason');
    const modeBadge = document.getElementById('mode-badge');
    const action = (dec.action || 'IDLE').toUpperCase();

    decEl.textContent = action;
    decEl.className = 'nn-decision ' + action.toLowerCase();
    reasonEl.textContent = dec.reason || '';
    modeBadge.textContent = action;
    modeBadge.className = 'badge mode-badge ' + action.toLowerCase();

    // Controller layers
    renderLayers('ctrl-layers', nn.controller ? nn.controller.layers : []);

    // Controller delta RPM bars
    if (nn.controller && nn.controller.output_delta) {
        renderBars('ctrl-delta', nn.controller.output_delta, 500);
    }

    // Lyapunov layers
    renderLayers('lyap-layers', nn.lyapunov ? nn.lyapunov.layers : []);

    // Lyapunov V value
    if (nn.lyapunov) {
        document.getElementById('nn-v-value').textContent = (nn.lyapunov.V_value || 0).toFixed(4);
    }

    // R singular values
    if (nn.lyapunov && nn.lyapunov.R_singular_values) {
        renderBars('lyap-sv', nn.lyapunov.R_singular_values, 1);
    }

    // SGD state
    if (nn.sgd) {
        document.getElementById('sgd-lr').textContent = nn.sgd.learning_rate.toExponential(0);
        document.getElementById('sgd-violation').textContent = nn.sgd.violation.toFixed(6);
        document.getElementById('sgd-grad-norm').textContent = nn.sgd.gradient_norm.toFixed(4);
        document.getElementById('sgd-latency').textContent = nn.sgd.latency_ms.toFixed(1) + ' ms';
        document.getElementById('sgd-adapt-count').textContent = nn.sgd.total_adaptations;
        const adaptedEl = document.getElementById('sgd-adapted');
        adaptedEl.textContent = nn.sgd.adapted ? 'YES' : 'No';
        adaptedEl.style.color = nn.sgd.adapted ? '#f59e0b' : '#64748b';
    }
}

function renderLayers(containerId, layers) {
    const container = document.getElementById(containerId);
    if (!container || !layers || layers.length === 0) return;

    let html = '';
    for (const layer of layers) {
        const name = layer.layer.replace('LeakyReLU', 'LReLU').replace('Softplus', 'SPlus');
        html += `<div class="nn-layer-row">
            <span class="nn-layer-name">${name}</span>
            <div class="nn-layer-viz">`;

        const vals = layer.values || [];
        const maxVal = Math.max(...vals.map(Math.abs), 0.01);
        for (let i = 0; i < Math.min(vals.length, 32); i++) {
            const norm = Math.abs(vals[i]) / maxVal;
            const hue = vals[i] >= 0 ? 220 : 0; // Blue positive, red negative
            const opacity = 0.15 + norm * 0.85;
            html += `<div class="nn-neuron" style="opacity:${opacity.toFixed(2)};background:hsl(${hue},80%,55%)"></div>`;
        }
        html += `</div></div>`;
    }
    container.innerHTML = html;
}

function renderBars(containerId, values, maxRef) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const maxVal = Math.max(...values.map(Math.abs), maxRef * 0.01);
    let html = '';
    for (const v of values) {
        const h = Math.abs(v) / maxVal * 28 + 2;
        const cls = v >= 0 ? 'positive' : 'negative';
        html += `<div class="nn-bar ${cls}" style="height:${h}px" title="${v.toFixed(2)}"></div>`;
    }
    container.innerHTML = html;
}

// -- Update Telemetry Panels ------------------------------------------
function updateTelemetry(status) {
    if (!status) return;

    // Lyapunov
    document.getElementById('lyap-value').textContent = (status.lyapunov_value || 0).toFixed(4);
    document.getElementById('roa-remaining').textContent = ((status.roa_remaining || 0) * 100).toFixed(1) + '%';

    const stabEl = document.getElementById('stability-status');
    if (status.is_stable) {
        stabEl.textContent = 'STABLE';
        stabEl.className = 'value status-ok';
    } else {
        stabEl.textContent = 'VIOLATION';
        stabEl.className = 'value status-danger';
    }

    // Safety
    const margin = status.safety_margin;
    document.getElementById('obstacle-dist').textContent = margin >= 99 ? 'Clear' : margin.toFixed(2) + ' m';
    document.getElementById('barrier-value').textContent =
        status.barrier_value >= 99 ? '--' : status.barrier_value.toFixed(3);

    // Update CBF section in NN panel
    document.getElementById('cbf-active').textContent = status.barrier_active ? 'YES' : 'No';
    document.getElementById('cbf-active').style.color = status.barrier_active ? '#ef4444' : '#64748b';
    document.getElementById('cbf-value').textContent =
        status.barrier_value >= 99 ? '--' : status.barrier_value.toFixed(4);
    document.getElementById('cbf-margin').textContent = margin >= 99 ? '--' : margin.toFixed(3) + ' m';

    // Motors
    if (status.motor_rpms) {
        for (let i = 0; i < 4; i++) {
            const rpm = status.motor_rpms[i] || 3980;
            document.getElementById(`motor${i + 1}-rpm`).textContent = Math.round(rpm);
            const bar = document.getElementById(`motor${i + 1}-bar`);
            bar.style.width = (rpm / 12000 * 100) + '%';
            bar.style.background = (rpm / 12000) > 0.8
                ? 'linear-gradient(90deg, #f59e0b, #ef4444)'
                : 'linear-gradient(90deg, #22c55e, #f59e0b)';
        }
    }

    // Battery
    const batt = status.battery_remaining || 100;
    document.getElementById('battery-pct').textContent = batt.toFixed(0) + '%';
    document.getElementById('battery-bar').style.width = batt + '%';

    // Position, velocity, goal
    if (status.position) {
        document.getElementById('position-val').textContent =
            `(${status.position[0].toFixed(1)}, ${status.position[1].toFixed(1)}, ${status.position[2].toFixed(1)})`;
    }
    if (status.velocity !== undefined) {
        document.getElementById('velocity-val').textContent = status.velocity.toFixed(2) + ' m/s';
    }
    if (status.goal_distance !== undefined) {
        document.getElementById('goal-dist-val').textContent = status.goal_distance.toFixed(2) + ' m';
    }
    document.getElementById('adapt-count').textContent = status.adaptations || 0;
    document.getElementById('replan-count').textContent = status.replans || 0;

    // Charts
    lyapHistory.push(status.lyapunov_value || 0);
    if (lyapHistory.length > MAX_HISTORY) lyapHistory.shift();
    drawMiniChart('chart-lyapunov', lyapHistory, '#3b82f6');

    barrierHistory.push(status.safety_margin < 99 ? status.safety_margin : 10);
    if (barrierHistory.length > MAX_HISTORY) barrierHistory.shift();
    drawMiniChart('chart-barrier', barrierHistory, '#22c55e', 0);
}

// -- Mini Chart -------------------------------------------------------
function drawMiniChart(canvasId, data, color, refLine) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (data.length < 2) return;

    const maxVal = Math.max(...data, 0.01);
    const minVal = Math.min(...data, 0);

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
        const x = (i / (data.length - 1)) * w;
        const y = h - ((data[i] - minVal) / (maxVal - minVal + 1e-6)) * h * 0.9 - 4;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = color + '15'; ctx.globalAlpha = 0.3; ctx.fill(); ctx.globalAlpha = 1;

    if (refLine !== undefined) {
        const ry = h - ((refLine - minVal) / (maxVal - minVal + 1e-6)) * h * 0.9 - 4;
        ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(0, ry); ctx.lineTo(w, ry); ctx.stroke(); ctx.setLineDash([]);
    }
}

// -- WebSocket + HTTP Polling Fallback ----------------------------------------
let connected = false;
let wsConnected = false;
let socket = null;

function handleTelemetryData(data) {
    if (!connected) {
        connected = true;
        document.getElementById('status-badge').textContent = 'CONNECTED';
        document.getElementById('status-badge').className = 'badge active';
    }
    if (data.running !== undefined) {
        simRunning = data.running;
        document.getElementById('btn-start').disabled = simRunning;
        document.getElementById('btn-stop').disabled = !simRunning;
    }
    if (data.state) updateDrone(data.state, data.status);
    if (data.status) updateTelemetry(data.status);
    if (data.neural_network) updateNNPanel(data.neural_network);
    if (data.math) updateMathPanel(data.math);
}

function initWebSocket() {
    try {
        socket = io({ transports: ['websocket', 'polling'] });
        socket.on('connect', () => {
            wsConnected = true;
            console.log('[WS] Connected via WebSocket');
            // Stop HTTP polling if WebSocket is active
            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }
        });
        socket.on('telemetry', handleTelemetryData);
        socket.on('disconnect', () => {
            wsConnected = false;
            console.log('[WS] Disconnected, falling back to HTTP polling');
            if (!pollInterval) {
                pollInterval = setInterval(pollServer, 100);
            }
        });
        socket.on('connect_error', () => {
            wsConnected = false;
            // Fallback to HTTP polling
            if (!pollInterval) {
                pollInterval = setInterval(pollServer, 100);
            }
        });
    } catch (e) {
        console.log('[WS] Socket.IO not available, using HTTP polling');
        pollInterval = setInterval(pollServer, 100);
    }
}

function pollServer() {
    fetch('/api/state')
        .then(res => res.json())
        .then(handleTelemetryData)
        .catch(() => {
            if (connected) {
                connected = false;
                document.getElementById('status-badge').textContent = 'OFFLINE';
                document.getElementById('status-badge').className = 'badge';
            }
        });
}

// -- Animation Loop ---------------------------------------------------
function animate() {
    requestAnimationFrame(animate);
    if (controls) controls.update();
    if (goalMarker) goalMarker.rotation.y += 0.02;
    renderer.render(scene, camera);
}

// -- Button Events ----------------------------------------------------
document.getElementById('btn-start').addEventListener('click', () => {
    fetch('/api/start', { method: 'POST' }).then(() => {
        simRunning = true;
        document.getElementById('btn-start').disabled = true;
        document.getElementById('btn-stop').disabled = false;
    }).catch(() => { });
});

document.getElementById('btn-stop').addEventListener('click', () => {
    fetch('/api/stop', { method: 'POST' }).then(() => {
        simRunning = false;
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-stop').disabled = true;
        // Fetch and show the simulation report
        fetchAndShowReport();
    }).catch(() => { });
});

document.getElementById('btn-reset').addEventListener('click', () => {
    fetch('/api/reset', { method: 'POST' }).then(() => {
        simRunning = false;
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-stop').disabled = true;
        drone.position.set(5, 5, 2);
        drone.quaternion.set(0, 0, 0, 1);
        drone.trail = [];
        lyapHistory = [];
        barrierHistory = [];
        clearObstacles3D();
    }).catch(() => { });
});

// -- Init -------------------------------------------------------------
initScene();
animate();
initWebSocket();
// Start HTTP polling as initial transport (WebSocket will take over when connected)
pollInterval = setInterval(pollServer, 100);
pollServer();

// -- Toggle Math Panel ------------------------------------------------
function toggleMathPanel() {
    const panel = document.getElementById('math-panel');
    if (panel) panel.classList.toggle('collapsed');
}

// -- Update Math Panel ------------------------------------------------
let _prevMath = {};

function updateMathPanel(math) {
    if (!math) return;

    const f = (v, d) => {
        if (v === undefined || v === null) return '\\text{--}';
        d = d !== undefined ? d : 4;
        return typeof v === 'number' ? v.toFixed(d) : String(v);
    };

    const numCls = (v) => v > 0 ? 'pos' : v < 0 ? 'neg' : 'neutral';
    const vec = (arr, d) => arr ? '[' + arr.map(x => typeof x === 'number' ? x.toFixed(d || 3) : x).join(', ') + ']' : '--';

    // Helper: render KaTeX safely
    function K(latex, displayMode) {
        try {
            return katex.renderToString(latex, { throwOnError: false, displayMode: !!displayMode });
        } catch (e) {
            return '<span class="math-eq-symbol">' + latex + '</span>';
        }
    }

    // Color helpers for KaTeX
    const cAcc = (s) => `\\textcolor{#60a5fa}{${s}}`;   // accent blue
    const cGrn = (s) => `\\textcolor{#4ade80}{${s}}`;   // green / positive
    const cRed = (s) => `\\textcolor{#f87171}{${s}}`;   // red / negative
    const cDim = (s) => `\\textcolor{#94a3b8}{${s}}`;   // dim / neutral
    const cYel = (s) => `\\textcolor{#fbbf24}{${s}}`;   // yellow / highlight
    const cVal = (v, d) => {
        const s = f(v, d);
        if (typeof v === 'number') return v >= 0 ? cGrn(s) : cRed(s);
        return s;
    };
    const cNum = (v, d) => cAcc(f(v, d));

    // ── CONTROLLER SECTION ──
    if (math.controller && math.controller.length > 0) {
        const steps = math.controller;
        const el = document.getElementById('ctrl-live-formula');
        const tl = document.getElementById('ctrl-timeline');
        if (!el || !tl) return;

        const inputStep = steps.find(s => s.label === 'Input State');
        const eqSubStep = steps.find(s => s.label === 'Equilibrium Subtraction');
        const clampStep = steps.find(s => s.label === 'Clamping');
        const layerSteps = steps.filter(s => s.label && s.label.startsWith('Layer'));
        const actSteps = steps.filter(s => s.label === 'Activation');

        // Live formula header
        if (eqSubStep && clampStep) {
            const uFinal = clampStep.u_final || [0, 0, 0, 0];
            el.innerHTML = K(`\\mathbf{u} = \\text{clamp}\\bigl(\\varphi_\\theta(\\xi) - \\varphi_\\theta(\\xi^*) + u^*,\\; u_{\\min},\\; u_{\\max}\\bigr)`, true) +
                '<div class="math-live-result">' +
                K(`= \\text{clamp}\\bigl(${cAcc(vec(eqSubStep.u_raw, 1))}\\,,\\; ${cDim('1000')},\\; ${cDim('12000')}\\bigr) = ${cYel(vec(uFinal, 1))} \\;\\text{RPM}`) +
                '</div>';
        }

        let html = '';

        // ① Input state
        if (inputStep && inputStep.values) {
            const p = inputStep.values['p (position)'] || [];
            const v = inputStep.values['v (velocity)'] || [];
            const q = inputStep.values['q (quaternion)'] || [];
            const w = inputStep.values['omega (ang vel)'] || [];
            const m = inputStep.values['motors (RPM)'] || [];
            html += `<div class="math-tl-step">
                <div class="math-tl-header"><span class="math-tl-label">① Input State</span></div>
                <div class="math-katex-block">${K(`\\xi = \\begin{bmatrix} \\mathbf{p} \\\\ \\mathbf{v} \\\\ \\mathbf{q} \\\\ \\boldsymbol{\\omega} \\\\ \\boldsymbol{\\Omega} \\end{bmatrix} = \\begin{bmatrix} ${cAcc(vec(p, 2))} \\\\ ${cAcc(vec(v, 3))} \\\\ ${cAcc(vec(q, 3))} \\\\ ${cAcc(vec(w, 4))} \\\\ ${cAcc(vec(m, 0))} \\end{bmatrix} \\in \\mathbb{R}^{17}`, true)}</div>
            </div>`;
        }

        // ② Forward Pass — show each layer
        if (layerSteps.length > 0) {
            html += `<div class="math-tl-step">
                <div class="math-tl-header"><span class="math-tl-label">② Forward Pass through MLP</span></div>`;
            for (let li = 0; li < layerSteps.length; li++) {
                const ls = layerSteps[li];
                const n = li + 1;
                if (ls.W_shape) {
                    html += `<div class="math-katex-block">${K(`\\mathbf{z}_{${n}} = W_{${n}}\\,\\mathbf{h}_{${n - 1}} + \\mathbf{b}_{${n}} \\quad \\bigl(W_{${n}} \\in \\mathbb{R}^{${ls.W_shape[0]}\\times${ls.W_shape[1]}},\\; \\|W_{${n}}\\| = ${cAcc(f(ls.W_norm, 3))}\\bigr)`)}</div>`;
                    if (ls.pre_activation) {
                        html += `<div class="math-katex-block">${K(`\\mathbf{z}_{${n}} = ${cAcc(vec(ls.pre_activation, 3))}`)}</div>`;
                    }
                }
            }
            // Show activations inline
            for (let i = 0; i < actSteps.length; i++) {
                const as = actSteps[i];
                const actName = as.desc === 'LeakyReLU' ? '\\text{LeakyReLU}' : `\\text{${as.desc}}`;
                html += `<div class="math-katex-block">${K(`\\mathbf{h}_{${i + 1}} = ${actName}(\\mathbf{z}_{${i + 1}}) = ${cAcc(vec(as.post_activation, 3))}`)}</div>`;
            }
            html += `</div>`;
        }

        // ④ Equilibrium subtraction
        if (eqSubStep) {
            html += `<div class="math-tl-step">
                <div class="math-tl-header"><span class="math-tl-label">③ Equilibrium Subtraction (Yang24 Eq.2)</span></div>
                <div class="math-katex-block">${K(`\\pi_\\theta(\\xi) = \\varphi(\\xi) - \\varphi(\\xi^*) + u^*`, true)}</div>
                <div class="math-katex-block">${K(`= ${cAcc(vec(eqSubStep.phi_xi, 3))} - ${cDim(vec(eqSubStep.phi_eq, 3))} + ${cDim(vec(eqSubStep.u_eq, 0))}`)}</div>
                <div class="math-katex-block">${K(`\\mathbf{u}_{\\text{raw}} = ${cYel(vec(eqSubStep.u_raw, 2))}`)}</div>
                <div class="math-eq-desc">Guarantees π(ξ*) = u* (hover RPMs at equilibrium)</div>
            </div>`;
        }

        // ⑤ Final clamped output
        if (clampStep) {
            const u = clampStep.u_final || [0, 0, 0, 0];
            html += `<div class="math-tl-step result">
                <div class="math-tl-header"><span class="math-tl-label">④ Final Motor Commands</span></div>
                <div class="math-katex-block">${K(`\\mathbf{u}_{\\text{final}} = \\text{clamp}\\bigl(\\mathbf{u}_{\\text{raw}},\\;${cDim('1000')},\\;${cDim('12000')}\\bigr)`, true)}</div>
                <div class="math-katex-block">${K(`= ${cYel(vec(u, 1))} \\;\\text{RPM}`)}</div>
            </div>`;
        }

        tl.innerHTML = html;
    }

    // ── LYAPUNOV SECTION ──
    if (math.lyapunov && math.lyapunov.length > 0) {
        const steps = math.lyapunov;
        const el = document.getElementById('lyap-live-formula');
        const tl = document.getElementById('lyap-timeline');
        if (!el || !tl) return;

        const totalStep = steps.find(s => s.label === 'Total Lyapunov Value');
        const decStep = steps.find(s => s.label === 'Decrease Condition');
        const nnStep = steps.find(s => s.label === 'NN Component');
        const linStep = steps.find(s => s.label === 'Quadratic-Norm Component');
        const deltaStep = steps.find(s => s.label === 'State Deviation');

        // Live formula header
        if (totalStep && decStep) {
            const stCol = decStep.is_stable ? cGrn : cRed;
            const stLabel = decStep.is_stable ? '\\; \\checkmark\\;\\text{STABLE}' : '\\; \\times\\;\\text{VIOLATION}';
            el.innerHTML = K(`V(\\xi) = \\underbrace{${cAcc(f(totalStep.V_nn, 4))}}_{V_{\\text{nn}}} + \\underbrace{${cAcc(f(totalStep.V_linear, 4))}}_{V_{\\text{lin}}} = ${cYel(f(totalStep.V_total, 4))}`, true) +
                '<div class="math-live-result">' +
                K(`\\dot{V} = V(\\xi^+) - (1-\\kappa)\\,V(\\xi) = ${stCol(f(decStep.V_dot, 6))} ${stCol(stLabel)}`) +
                '</div>';
        }

        let html = '';

        // ① State deviation
        if (deltaStep) {
            html += `<div class="math-tl-step">
                <div class="math-tl-header"><span class="math-tl-label">① State Deviation from Equilibrium</span></div>
                <div class="math-katex-block">${K(`\\boldsymbol{\\delta} = \\xi - \\xi^* , \\quad \\|\\boldsymbol{\\delta}\\| = ${cAcc(f(deltaStep.delta_norm, 4))}`, true)}</div>
                <div class="math-katex-block">${K(`\\text{Top-5 } |\\delta_i|: \\; ${cAcc(vec(deltaStep.delta_top5, 4))}`)}</div>
            </div>`;
        }

        // ② NN component
        if (nnStep) {
            html += `<div class="math-tl-step">
                <div class="math-tl-header"><span class="math-tl-label">② Neural Network Component</span></div>
                <div class="math-katex-block">${K(`V_{\\text{nn}}(\\xi) = \\bigl\\| \\phi_V(\\xi) - \\phi_V(\\xi^*) \\bigr\\|_1`, true)}</div>
                <div class="math-katex-block">${K(`= \\bigl| \\underbrace{${cAcc(f(nnStep.phi_xi, 4))}}_{\\phi_V(\\xi)} - \\underbrace{${cDim(f(nnStep.phi_eq, 4))}}_{\\phi_V(\\xi^*)} \\bigr| = ${cYel(f(nnStep.V_nn, 4))}`)}</div>
                <div class="math-eq-desc">φ_V : ℝ¹⁷ → ℝ¹ via MLP [17→32→16→1] with Tanh + Softplus</div>
            </div>`;
        }

        // ③ Quadratic-norm component
        if (linStep) {
            html += `<div class="math-tl-step">
                <div class="math-tl-header"><span class="math-tl-label">③ Quadratic-Norm Component</span></div>
                <div class="math-katex-block">${K(`V_{\\text{lin}}(\\xi) = \\bigl\\| (\\varepsilon I + R^\\top R)\\,\\boldsymbol{\\delta} \\bigr\\|_1`, true)}</div>
                <div class="math-katex-block">${K(`R = U\\,\\text{diag}\\bigl(\\text{softplus}(\\sigma) + \\psi^2\\bigr)\\,V_h^\\top`)}</div>
                <div class="math-katex-block">${K(`\\varepsilon = ${cDim(f(linStep.epsilon, 4))}, \\quad \\kappa(R) = \\frac{\\sigma_{\\max}}{\\sigma_{\\min}} = ${cAcc(f(linStep.R_condition, 2))}`)}</div>
                <div class="math-katex-block">${K(`\\sigma(R) = ${cAcc(vec(linStep.R_singular_values, 3))}`)}</div>
                <div class="math-katex-block">${K(`V_{\\text{lin}} = ${cYel(f(linStep.V_linear, 4))}`)}</div>
            </div>`;
        }

        // ④ Total
        if (totalStep) {
            html += `<div class="math-tl-step result">
                <div class="math-tl-header"><span class="math-tl-label">④ Total Lyapunov Value</span></div>
                <div class="math-katex-block">${K(`V(\\xi) = V_{\\text{nn}} + V_{\\text{lin}} = ${cAcc(f(totalStep.V_nn, 4))} + ${cAcc(f(totalStep.V_linear, 4))} = \\boxed{${cYel(f(totalStep.V_total, 4))}}`, true)}</div>
                <div class="math-eq-desc">V(ξ) > 0 ∀ ξ ≠ ξ* guaranteed by εI + R⊤R ≻ 0</div>
            </div>`;
        }

        // ⑤ Decrease condition
        if (decStep) {
            const cls = decStep.is_stable ? 'result' : 'danger';
            const stCol = decStep.V_dot <= 0 ? cGrn : cRed;
            html += `<div class="math-tl-step ${cls}">
                <div class="math-tl-header">
                    <span class="math-tl-label">⑤ Lyapunov Decrease Condition</span>
                    <span class="math-tl-badge ${decStep.is_stable ? 'stable' : 'unstable'}">${decStep.is_stable ? 'STABLE' : 'VIOLATION'}</span>
                </div>
                <div class="math-katex-block">${K(`\\Delta V = V(\\xi^+) - (1-\\kappa)\\,V(\\xi) \\leq 0 \\quad \\text{(required for stability)}`, true)}</div>
                <div class="math-katex-block">${K(`= ${cAcc(f(decStep.V_next, 4))} - (1 - ${cDim(f(decStep.kappa, 2))}) \\times ${cAcc(f(decStep.V_curr, 4))}`)}</div>
                <div class="math-katex-block">${K(`= ${cAcc(f(decStep.V_next, 4))} - ${cDim(f((1 - decStep.kappa) * decStep.V_curr, 4))}`)}</div>
                <div class="math-katex-block">${K(`\\boxed{\\Delta V = ${stCol(f(decStep.V_dot, 6))}} \\quad ${decStep.is_stable ? cGrn('\\checkmark \\; \\leq 0') : cRed('\\times \\; > 0')}`, true)}</div>
            </div>`;
        }

        tl.innerHTML = html;
    }

    // ── BARRIER SECTION ──
    {
        const el = document.getElementById('barrier-live-formula');
        const tl = document.getElementById('barrier-timeline');
        if (!el || !tl) return;

        if (math.barrier && math.barrier.length > 0) {
            // Header summary
            el.innerHTML = math.barrier.map((obs, i) => {
                const col = obs.is_safe ? cGrn : cRed;
                return K(`B_{${i + 1}} = ${col(f(obs.B_value, 3))} \\; ${obs.is_safe ? cGrn('\\checkmark') : cRed('\\times')}`);
            }).join('&nbsp;&nbsp;&nbsp;');

            let html = '';
            for (let i = 0; i < math.barrier.length; i++) {
                const obs = math.barrier[i];
                const cls = obs.is_safe ? '' : 'danger';
                const oc = obs.obstacle_center || [0, 0, 0];
                const pos = obs.position || [0, 0, 0];
                const distSq = ((pos[0] - oc[0]) ** 2 + (pos[1] - oc[1]) ** 2 + (pos[2] - oc[2]) ** 2);
                const rSafeSq = (obs.r_safe || 0) ** 2;

                html += `<div class="math-tl-step ${cls}">
                    <div class="math-tl-header">
                        <span class="math-tl-label">Obstacle ${i + 1}</span>
                        <span class="math-tl-badge ${obs.is_safe ? 'safe' : 'unsafe'}">${obs.is_safe ? 'SAFE' : 'DANGER'}</span>
                    </div>
                    <div class="math-katex-block">${K(`B(\\xi) = \\|\\mathbf{p} - \\mathbf{o}\\|^2 - r_{\\text{safe}}^2`, true)}</div>
                    <div class="math-katex-block">${K(`= \\bigl\\| ${cAcc(vec(pos, 2))} - ${cDim(vec(oc, 2))} \\bigr\\|^2 - ${cDim(f(obs.r_safe, 2))}^2`)}</div>
                    <div class="math-katex-block">${K(`= ${cAcc(f(distSq, 3))} - ${cDim(f(rSafeSq, 3))} = ${(obs.is_safe ? cGrn : cRed)(f(obs.B_value, 4))}`)}</div>
                    <div class="math-katex-block">${K(`\\|\\mathbf{p}-\\mathbf{o}\\| = ${cAcc(f(obs.distance, 3))}\\,\\text{m}, \\quad r_{\\text{safe}} = r_{\\text{obs}} + r_{\\text{drone}} = ${cDim(f(obs.r_safe, 3))}\\,\\text{m}`)}</div>
                    <div class="math-katex-block">${K(`B \\geq 0 \\iff \\text{safe} \\quad \\Rightarrow \\quad ${(obs.is_safe ? cGrn('\\text{SAFE}') : cRed('\\text{UNSAFE}'))}`)}</div>
                </div>`;
            }
            tl.innerHTML = html;
        } else {
            el.innerHTML = K(`B(\\xi) = \\|\\mathbf{p} - \\mathbf{o}\\|^2 - r_{\\text{safe}}^2 \\quad \\text{(no obstacles — inactive)}`);
            tl.innerHTML = '';
        }
    }

    // ── DYNAMICS SECTION ──
    if (math.dynamics) {
        const d = math.dynamics;
        const el = document.getElementById('dyn-live-formula');
        const tl = document.getElementById('dyn-timeline');
        if (!el || !tl) return;

        const tw = d.thrust_to_weight || 0;
        const twCol = (tw >= 0.9 && tw <= 1.1) ? cGrn : cRed;
        el.innerHTML = K(`\\xi_{t+1} = f(\\xi_t, \\mathbf{u}_t), \\quad \\frac{T}{W} = \\frac{${cAcc(f(d.total_thrust_N, 3))}}{${cDim(f(d.gravity_force_N, 3))}} = ${twCol(f(tw, 4))}`, true);

        let html = '';
        const rpms = d.motor_rpms || [];
        const omegas = d.omega_rad_s || [];
        const thrusts = d.thrust_per_motor_N || [];

        // ① Motor conversion
        html += `<div class="math-tl-step">
            <div class="math-tl-header"><span class="math-tl-label">① RPM → Angular Velocity</span></div>
            <div class="math-katex-block">${K(`\\omega_i = \\Omega_i \\times \\frac{2\\pi}{60} \\quad \\text{[rad/s]}`, true)}</div>
            <div class="math-katex-block">${K(`\\boldsymbol{\\Omega} = ${cAcc(vec(rpms.map(r => Math.round(r)), 0))} \\;\\text{RPM}`)}</div>
            <div class="math-katex-block">${K(`\\boldsymbol{\\omega} = ${cAcc(vec(omegas, 2))} \\;\\text{rad/s}`)}</div>
        </div>`;

        // ② Thrust
        html += `<div class="math-tl-step">
            <div class="math-tl-header"><span class="math-tl-label">② Thrust Generation</span></div>
            <div class="math-katex-block">${K(`T_i = k_f \\cdot \\omega_i^2, \\quad k_f = ${cDim('1.2 \\times 10^{-5}')} \\; \\text{N/(rad/s)}^2`, true)}</div>
            <div class="math-katex-block">${K(`\\mathbf{T} = ${cAcc(vec(thrusts, 4))} \\;\\text{N}`)}</div>
            <div class="math-katex-block">${K(`F_z = \\sum_i T_i = ${cYel(f(d.total_thrust_N, 4))} \\;\\text{N}`)}</div>
        </div>`;

        // ③ Torques
        html += `<div class="math-tl-step">
            <div class="math-tl-header"><span class="math-tl-label">③ Torque Allocation (X-config)</span></div>
            <div class="math-katex-block">${K(`\\tau_{\\text{roll}} = L(T_2 - T_4), \\quad \\tau_{\\text{pitch}} = L(T_3 - T_1)`, true)}</div>
            <div class="math-katex-block">${K(`\\tau_{\\text{yaw}} = k_m(-\\omega_1^2 + \\omega_2^2 - \\omega_3^2 + \\omega_4^2)`)}</div>
            <div class="math-eq-desc">L = arm length = 0.18 m, k_m = 1.5×10⁻⁷ N·m/(rad/s)²</div>
        </div>`;

        // ④ Equations of motion
        html += `<div class="math-tl-step">
            <div class="math-tl-header"><span class="math-tl-label">④ Equations of Motion (Euler Integration)</span></div>
            <div class="math-katex-block">${K(`\\mathbf{p}_{t+1} = \\mathbf{p}_t + \\mathbf{v}_t \\cdot \\Delta t`, true)}</div>
            <div class="math-katex-block">${K(`\\mathbf{v}_{t+1} = \\mathbf{v}_t + \\bigl(\\mathbf{g} + R\\,\\frac{F_z}{m}\\hat{z} + \\mathbf{a}_{\\text{drag}}\\bigr) \\cdot \\Delta t`)}</div>
            <div class="math-katex-block">${K(`\\mathbf{q}_{t+1} = \\mathbf{q}_t + \\tfrac{1}{2}\\,\\Omega(\\omega)\\,\\mathbf{q}_t \\cdot \\Delta t`)}</div>
            <div class="math-katex-block">${K(`\\boldsymbol{\\omega}_{t+1} = \\boldsymbol{\\omega}_t + J^{-1}(\\boldsymbol{\\tau} - \\boldsymbol{\\omega} \\times J\\boldsymbol{\\omega}) \\cdot \\Delta t`)}</div>
        </div>`;

        // ⑤ Force balance result
        html += `<div class="math-tl-step result">
            <div class="math-tl-header"><span class="math-tl-label">⑤ Force Balance</span></div>
            <div class="math-katex-block">${K(`\\frac{\\sum T_i}{mg} = \\frac{${cAcc(f(d.total_thrust_N, 4))}}{${cDim(f(d.gravity_force_N, 4))}} = \\boxed{${twCol(f(tw, 4))}}`, true)}</div>
            <div class="math-eq-desc">Δt = ${d.dt}s (100 Hz). Ideal hover: T/W = 1.0</div>
        </div>`;

        tl.innerHTML = html;
    }

    _prevMath = math;
}

// -- Post-Simulation Report -------------------------------------------
function fetchAndShowReport() {
    fetch('/api/report')
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                console.log("Report error:", data.error);
                return;
            }
            populateReportModal(data);
        })
        .catch(err => console.error("Failed to fetch report:", err));
}

function populateReportModal(data) {
    // 1. Images
    const imgContainer = document.getElementById('report-images');
    imgContainer.innerHTML = '';
    if (data.images && data.images.length > 0) {
        data.images.forEach(url => {
            const img = document.createElement('img');
            img.src = url + "?t=" + new Date().getTime();
            imgContainer.appendChild(img);
        });
    }

    const t = (k, v, color) => `<tr><td style="padding:4px;border-bottom:1px dashed rgba(255,255,255,0.05);color:#94a3b8;">${k}</td><td style="text-align:right;padding:4px;border-bottom:1px dashed rgba(255,255,255,0.05);color:${color||'#e2e8f0'};font-family:monospace;">${v}</td></tr>`;
    const beginTable = `<table style="width:100%; border-collapse: collapse; font-size:11px;">`;
    const endTable = `</table>`;

    // 2. Controller
    let ctrlHtml = beginTable;
    if (data.math && data.math.controller) {
        let cv = data.math.controller;
        let cclamp = cv.find(s => s.label === "Clamping");
        let ceq = cv.find(s => s.label === "Equilibrium Subtraction");
        
        if (ceq && ceq.u_raw && ceq.phi_xi) {
            let p1 = "[" + ceq.phi_xi.map(r=>Math.round(r)).join(", ") + "]";
            let p2 = "[" + ceq.phi_eq.map(r=>Math.round(r)).join(", ") + "]";
            let p3 = "[" + ceq.u_eq.map(r=>Math.round(r)).join(", ") + "]";
            ctrlHtml += t("u_raw = &phi;(&xi;) - &phi;(&xi;*) + u*", p1 + " - " + p2 + " + " + p3);
            ctrlHtml += t("      =", "[" + ceq.u_raw.map(r=>Math.round(r)).join(", ") + "]");
        } else if (ceq && ceq.u_raw) {
            ctrlHtml += t("Raw NN Output + Ref", "[" + ceq.u_raw.map(r=>Math.round(r)).join(", ") + "]");
        }
        if (cclamp && cclamp.u_final) {
             let raw = ceq ? "[" + ceq.u_raw.map(r=>Math.round(r)).join(",") + "]" : "u_raw";
             ctrlHtml += t("u = clamp(u_raw, 1000, 12000)", "clamp(" + raw + ")", "#fbbf24");
             ctrlHtml += t("  =", "[" + cclamp.u_final.map(r=>Math.round(r)).join(", ") + "]", "#fbbf24");
        }
    }
    document.getElementById('report-ctrl').innerHTML = ctrlHtml + endTable;

    // 3. Lyapunov
    let lyapHtml = beginTable;
    if (data.math && data.math.lyapunov) {
        let lv = data.math.lyapunov;
        let ltot = lv.find(s => s.label === "Total Lyapunov Value");
        let ldec = lv.find(s => s.label === "Decrease Condition");
        let lnn = lv.find(s => s.label === "NN Component");
        let llin = lv.find(s => s.label === "Quadratic-Norm Component");
        
        if (lnn && lnn.V_nn !== undefined) lyapHtml += t("V_nn", lnn.V_nn.toFixed(4));
        if (llin && llin.V_linear !== undefined) lyapHtml += t("V_lin", llin.V_linear.toFixed(4));
        
        if (ltot && ltot.V_total !== undefined && lnn && llin) {
            lyapHtml += t("Total V(&xi;) = V_nn + V_lin", `${lnn.V_nn.toFixed(4)} + ${llin.V_linear.toFixed(4)}`);
            lyapHtml += t("           =", ltot.V_total.toFixed(4), "#60a5fa");
        }
        else if (ltot && ltot.V_total !== undefined) lyapHtml += t("Total V(&xi;)", ltot.V_total.toFixed(4), "#60a5fa");
        
        if (ldec) {
            let col = ldec.V_dot <= 0 ? "#4ade80" : "#f87171";
            lyapHtml += t("&Delta;V = V(&xi;&spplus;) - (1-&kappa;)V(&xi;)", `${ldec.V_next.toFixed(4)} - ${(1 - ldec.kappa).toFixed(2)} &times; ${ldec.V_curr.toFixed(4)}`);
            lyapHtml += t("   =", ldec.V_dot.toFixed(6) + " " + (ldec.is_stable ? "✓" : "✗"), col);
        }
    }
    document.getElementById('report-lyap').innerHTML = lyapHtml + endTable;

    // 4. Barrier
    let barHtml = beginTable;
    if (data.math && data.math.barrier && data.math.barrier.length > 0) {
        let obs = data.math.barrier[0];
        let pStr = "[" + (obs.position || [0,0,0]).map(x=>x.toFixed(1)).join(", ") + "]";
        let oStr = "[" + (obs.obstacle_center || [0,0,0]).map(x=>x.toFixed(1)).join(", ") + "]";
        barHtml += t("B(&xi;) = ||p - o||&sup2; - r_safe&sup2;", `||${pStr} - ${oStr}||&sup2; - ${obs.r_safe.toFixed(3)}&sup2;`);
        
        let col = obs.is_safe ? "#4ade80" : "#f87171";
        barHtml += t("     =", obs.B_value.toFixed(4) + " " + (obs.is_safe ? "✓" : "✗"), col);
    } else {
        barHtml += t("Status", "No active constraints", "#94a3b8");
    }
    document.getElementById('report-barr').innerHTML = barHtml + endTable;

    // 5. Neural Architecture
    let archHtml = beginTable;
    if (data.architecture) {
        let a = data.architecture;
        if (a.controller) {
            archHtml += `<tr><td colspan="2" style="color:#4ade80; padding-top:8px; padding-bottom:4px; font-weight:bold;">Controller Net</td></tr>`;
            archHtml += t("Structure", a.controller.architecture);
        }
        if (a.lyapunov) {
            archHtml += `<tr><td colspan="2" style="color:#4ade80; padding-top:8px; padding-bottom:4px; font-weight:bold;">Lyapunov Net</td></tr>`;
            archHtml += t("Structure", a.lyapunov.architecture);
        }
    }
    document.getElementById('report-arch').innerHTML = archHtml + endTable;

    // 6. Adaptive SGD
    let sgdHtml = beginTable;
    sgdHtml += t("Total Simulation Steps", data.steps);
    sgdHtml += t("Flight Duration", data.duration.toFixed(2) + "s");
    sgdHtml += t("SGD Adaptations Triggered", data.sgd_adaptations || 0, (data.sgd_adaptations > 0 ? "#fbbf24" : null));
    
    if (data.architecture && data.architecture.sgd) {
        let s = data.architecture.sgd;
        sgdHtml += t("Learning Rate", s.learning_rate);
        sgdHtml += t("Current Violation", (s.violation !== undefined ? s.violation.toFixed(5) : "0.00000"));
        sgdHtml += t("Gradient Norm", (s.gradient_norm !== undefined ? s.gradient_norm.toFixed(5) : "0.00000"));
    }
    document.getElementById('report-sgd').innerHTML = sgdHtml + endTable;

    document.getElementById('report-modal').classList.add('show');
}

// Close modal handler
document.getElementById('close-report-btn').addEventListener('click', () => {
    document.getElementById('report-modal').classList.remove('show');
});

// Close when clicking outside
window.addEventListener('click', (e) => {
    const rModal = document.getElementById('report-modal');
    const eModal = document.getElementById('explain-modal');
    if (e.target === rModal) rModal.classList.remove('show');
    if (e.target === eModal) eModal.classList.remove('show');
});

// --- Decision Explainer Logic ---
document.getElementById('btn-explain').addEventListener('click', () => {
    const modal = document.getElementById('explain-modal');
    const badge = document.getElementById('explain-action-badge');
    const reasonText = document.getElementById('explain-reason');
    const tlContainer = document.getElementById('explain-timeline');
    
    const modeBadge = document.getElementById('mode-badge');
    badge.textContent = modeBadge.textContent;
    badge.className = modeBadge.className;
    
    // Find nn-reason if it exists
    const nnReason = document.getElementById('nn-reason');
    reasonText.textContent = nnReason ? nnReason.textContent : "No reason logged.";
    
    tlContainer.innerHTML = '';
    
    if (typeof _prevMath === 'undefined' || !_prevMath) {
        tlContainer.innerHTML = '<div style="color:var(--text-dim); padding: 20px;">No telemetry data yet. Run the simulation first.</div>';
        modal.classList.add('show');
        return;
    }
    
    let html = '';
    const m = _prevMath;
    
    // 1. Controller Output
    if (m.controller) {
        let ceq = m.controller.find(s => s.label === "Equilibrium Subtraction");
        let u_final = m.controller.find(s => s.label === "Clamping")?.u_final || [0,0,0,0];
        
        let subStr = ceq && ceq.u_raw ? `[${ceq.u_raw.map(r=>Math.round(r)).join(',')}]` : "N/A";
        html += `<div class="explain-step success">
            <div class="explain-step-title"><span>1. Controller Forward Pass</span> <span>(Neural Network)</span></div>
            <div class="explain-step-desc">The controller MLP computed the base target RPMs based on the current position error and velocity.</div>
            <div class="explain-step-math">u_raw = ${subStr} <br> u_out = [${u_final.map(r=>Math.round(r)).join(',')}]</div>
        </div>`;
    }
    
    // 2. Lyapunov Check
    if (m.lyapunov) {
        let dec = m.lyapunov.find(s => s.label === "Decrease Condition");
        let V_val = m.lyapunov.find(s => s.label === "Total Lyapunov Value")?.V_total || 0;
        
        if (dec) {
            let cl = dec.is_stable ? "success" : "danger";
            html += `<div class="explain-step ${cl}">
                <div class="explain-step-title"><span>2. Stability Certificate</span> <span>(Lyapunov Critic)</span></div>
                <div class="explain-step-desc">The Lyapunov network verified if the controller's action will stabilize the drone. A valid step requires &Delta;V &le; 0.</div>
                <div class="explain-step-math">V(&xi;) = ${V_val.toFixed(4)} <br> &Delta;V = ${dec.V_dot.toFixed(6)} ${dec.is_stable ? "✓ Safe" : "✗ Violation"}</div>
            </div>`;
        }
    }
    
    // 3. CBF Check
    if (m.barrier && m.barrier.length > 0) {
        let obs = m.barrier[0];
        let cl = obs.is_safe ? "success" : "warning";
        html += `<div class="explain-step ${cl}">
            <div class="explain-step-title"><span>3. Safety Filter</span> <span>(Barrier Function)</span></div>
            <div class="explain-step-desc">The Control Barrier Function (CBF) checked for imminent collisions. If B(&xi;) < 0, it intervenes to project the control commands into a safe set.</div>
            <div class="explain-step-math">Distance to obstacle: ${obs.distance.toFixed(2)}m <br> B(&xi;) = ${obs.B_value.toFixed(4)} ${obs.is_safe ? "✓ Safe" : "⚠ Intervening"}</div>
        </div>`;
    } else {
        html += `<div class="explain-step">
            <div class="explain-step-title"><span>3. Safety Filter</span> <span>(Barrier Function)</span></div>
            <div class="explain-step-desc">No obstacles detected within the safety horizon. The barrier function was inactive.</div>
        </div>`;
    }
    
    // 4. Adaptation Result
    const isAdapting = modeBadge.classList.contains('adapting');
    if (isAdapting) {
        html += `<div class="explain-step warning">
            <div class="explain-step-title"><span>4. Online Learning</span> <span>(Stability-Aware SGD)</span></div>
            <div class="explain-step-desc">Because the Lyapunov stability condition (&Delta;V &le; 0) was violated, the system paused physically applying the nominal action and triggered an online gradient descent step to adapt the Controller MLP weights in real-time.</div>
        </div>`;
    } else {
        html += `<div class="explain-step success">
            <div class="explain-step-title"><span>4. Action Applied</span> <span>(Dynamics Engine)</span></div>
            <div class="explain-step-desc">The final RPMs were deemed safe and stabilizing, and were sent directly to the four motors.</div>
        </div>`;
    }
    
    tlContainer.innerHTML = html;
    modal.classList.add('show');
});

document.getElementById('close-explain-btn').addEventListener('click', () => {
    document.getElementById('explain-modal').classList.remove('show');
});

