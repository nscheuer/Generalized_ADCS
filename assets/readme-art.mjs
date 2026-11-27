// Draws the README banner and architecture diagram in light and dark versions.
// Usage (from the repository root): node assets/readme-art.mjs assets
// Colours follow docs/source/_static/custom.css (#0066cc light / #80c0ff dark).
import fs from 'node:fs';
import path from 'node:path';

const OUT = process.argv[2] ?? 'assets';
fs.mkdirSync(OUT, { recursive: true });

const THEMES = {
  light: {
    bg: '#fbfcfe', code: '#f4f7fb', fg: '#14202e', fgSoft: '#4f5d6e', faint: '#8a96a6',
    line: '#dfe5ee', accent: '#0066cc', accentSoft: '#e8f1fc', orbit: '#b9cde6',
    ok: '#1a7f4b', okSoft: '#e7f6ed', warm: '#a35a00', warmSoft: '#fdf1e3',
    key: '#8129a8', fn: '#0052a3', str: '#0a7a5e', num: '#9a5300', com: '#8a96a6',
  },
  dark: {
    bg: '#0f1620', code: '#162130', fg: '#e5ecf4', fgSoft: '#a3b1c2', faint: '#6f7e91',
    line: '#253244', accent: '#80c0ff', accentSoft: '#15263b', orbit: '#2c4463',
    ok: '#5cc98d', okSoft: '#13291e', warm: '#f0b46c', warmSoft: '#2b2114',
    key: '#d79bf0', fn: '#80c0ff', str: '#6fd0ae', num: '#e0a870', com: '#6f7e91',
  },
};

const SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Inter, Roboto, Helvetica, Arial, sans-serif";
const MONO = "ui-monospace, 'SF Mono', 'Cascadia Code', Menlo, Consolas, 'Liberation Mono', monospace";
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/* Minimal Python highlighter, enough for the banner snippet. */
function highlight(line, t) {
  const KEYS = new Set(['import', 'from', 'as', 'for', 'in']);
  const re = /(#.*$)|("[^"]*")|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)|([A-Za-z_][\w]*)(?=\()|([A-Za-z_][\w]*)|(\s+)|(.)/g;
  let out = '', m;
  while ((m = re.exec(line))) {
    const [s, com, str, num, call, word] = m;
    const color = com ? t.com : str ? t.str : num ? t.num : call ? t.fn
      : word ? (KEYS.has(word) ? t.key : t.fg) : t.faint;
    out += `<tspan fill="${color}">${esc(s)}</tspan>`;
  }
  return out;
}

function banner(t) {
  // Lines taken from the README quickstart, shortened for display.
  const code = [
    'import ADCS',
    '',
    'sat = ADCS.Satellite(J_0=J, actuators=acts,',
    '                     sensors=sens)',
    'ctrl = ADCS.controller.MTQ_w_RW_LP(est_sat=sat, ...)',
    'goal = ADCS.goals.Coordinate_Goal(lat=42.36,',
    '                                  lon=-71.06, alt=0)',
    'results = ADCS.simulate(x=x_0, satellite=sat,',
    '    controller=ctrl, goal=goal, os0=os0, tf=500.0)',
  ];
  const cx = 690, cy = 46, cw = 556, ch = 288;
  const lines = code.map((l, i) =>
    `<text x="${cx + 24}" y="${cy + 78 + i * 24}" font-family="${MONO}" font-size="15.5" xml:space="preserve">${highlight(l, t)}</text>`).join('\n  ');

  // Earth, an inclined orbit and a small spacecraft with a boresight arrow.
  const ex = 116, ey = 292;
  const motif = `
  <ellipse cx="${ex}" cy="${ey}" rx="250" ry="70" transform="rotate(-14 ${ex} ${ey})" fill="none" stroke="${t.orbit}" stroke-width="1.6" stroke-dasharray="5 6"/>
  <circle cx="${ex}" cy="${ey}" r="46" fill="${t.accentSoft}" stroke="${t.accent}" stroke-width="1.4"/>
  <path d="M${ex - 46} ${ey} Q ${ex} ${ey - 16} ${ex + 46} ${ey}" fill="none" stroke="${t.accent}" stroke-opacity="0.45"/>
  <path d="M${ex - 40} ${ey + 22} Q ${ex} ${ey + 8} ${ex + 40} ${ey + 22}" fill="none" stroke="${t.accent}" stroke-opacity="0.35"/>
  <g transform="translate(340 236) rotate(-24)">
    <rect x="-9" y="-9" width="18" height="18" rx="2" fill="${t.accent}"/>
    <rect x="-34" y="-5" width="20" height="10" rx="1.5" fill="none" stroke="${t.accent}" stroke-width="1.4"/>
    <rect x="14" y="-5" width="20" height="10" rx="1.5" fill="none" stroke="${t.accent}" stroke-width="1.4"/>
  </g>
  <path d="M332 252 L 214 286" stroke="${t.accent}" stroke-width="1.6" stroke-dasharray="3 4"/>`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="380" viewBox="0 0 1280 380" role="img" aria-label="Generalized ADCS: spacecraft attitude determination and control in Python">
  <defs>
    <pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse">
      <circle cx="2" cy="2" r="1.1" fill="${t.line}"/>
    </pattern>
    <linearGradient id="fade" x1="0" x2="1" y1="0" y2="0">
      <stop offset="0" stop-color="${t.bg}" stop-opacity="1"/>
      <stop offset="0.5" stop-color="${t.bg}" stop-opacity="0.25"/>
      <stop offset="1" stop-color="${t.bg}" stop-opacity="0"/>
    </linearGradient>
    <clipPath id="frame"><rect width="1280" height="380" rx="18"/></clipPath>
  </defs>
  <g clip-path="url(#frame)">
    <rect width="1280" height="380" fill="${t.bg}"/>
    <rect width="1280" height="380" fill="url(#dots)"/>
    <rect width="1280" height="380" fill="url(#fade)"/>
    ${motif}
  </g>
  <rect x="0.5" y="0.5" width="1279" height="379" rx="18" fill="none" stroke="${t.line}"/>

  <text x="64" y="116" font-family="${SANS}" font-size="58" font-weight="700" letter-spacing="-2" fill="${t.fg}">Generalized ADCS</text>
  <text x="67" y="164" font-family="${SANS}" font-size="25" font-weight="600" letter-spacing="-0.3" fill="${t.fg}">Spacecraft attitude determination</text>
  <text x="67" y="196" font-family="${SANS}" font-size="25" font-weight="600" letter-spacing="-0.3" fill="${t.fg}">and control in <tspan fill="${t.accent}">Python</tspan></text>

  <rect x="${cx}" y="${cy}" width="${cw}" height="${ch}" rx="12" fill="${t.code}" stroke="${t.line}"/>
  <path d="M${cx} ${cy + 42}h${cw}" stroke="${t.line}"/>
  <circle cx="${cx + 22}" cy="${cy + 21}" r="5.5" fill="${t.line}"/>
  <circle cx="${cx + 40}" cy="${cy + 21}" r="5.5" fill="${t.line}"/>
  <circle cx="${cx + 58}" cy="${cy + 21}" r="5.5" fill="${t.line}"/>
  <text x="${cx + cw - 22}" y="${cy + 26}" text-anchor="end" font-family="${MONO}" font-size="13.5" fill="${t.faint}">quickstart.py</text>
  ${lines}
</svg>
`;
}

function diagram(t) {
  const W = 1280, H = 470;
  const box = (x, y, w, h, title, sub, { fill = t.code, stroke = t.line, color = t.fg } = {}) => `
  <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="${fill}" stroke="${stroke}"/>
  <text x="${x + 18}" y="${y + 32}" font-family="${SANS}" font-size="19" font-weight="650" fill="${color}">${esc(title)}</text>
  ${sub.map((s, i) => `<text x="${x + 18}" y="${y + 56 + i * 21}" font-family="${SANS}" font-size="14" fill="${t.fgSoft}">${s}</text>`).join('\n  ')}`;
  const arrow = (d, x2, y2, a, color = t.faint) => {
    const L = 9, p = (s) => `${x2 - L * Math.cos(a + s)},${y2 - L * Math.sin(a + s)}`;
    return `<path d="${d}" stroke="${color}" stroke-width="1.8" fill="none"/>
  <path d="M${x2} ${y2} L${p(0.45)} L${p(-0.45)} Z" fill="${color}"/>`;
  };
  const label = (x, y, s, anchor = 'middle') =>
    `<text x="${x}" y="${y}" text-anchor="${anchor}" font-family="${SANS}" font-size="12.5" fill="${t.faint}">${s}</text>`;
  const mono = (s) => `<tspan font-family="${MONO}" font-size="13.5" fill="${t.accent}">${s}</tspan>`;
  const R = 0, D = Math.PI / 2, Lf = Math.PI, U = -Math.PI / 2;

  // Closed loop: Satellite -> sensors -> Estimator -> Controller -> actuators -> Satellite
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="How a Generalized ADCS simulation fits together: ADCS.simulate propagates the orbit and environment and closes the loop from the satellite's sensors, through an attitude estimator and a controller that tracks a goal, back to the actuators. The results feed plots, the Godot viewer and Monte Carlo campaigns.">
  <rect width="${W}" height="${H}" rx="18" fill="${t.bg}"/>
  <rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" rx="18" fill="none" stroke="${t.line}"/>

  <rect x="24" y="24" width="860" height="422" rx="14" fill="none" stroke="${t.accent}" stroke-dasharray="6 6" stroke-opacity="0.7"/>
  <text x="44" y="56" font-family="${MONO}" font-size="17" font-weight="600" fill="${t.accent}">ADCS.simulate(...)</text>

  ${box(40, 80, 236, 150, 'Orbit & environment', [`${mono('Orbital_State')}, ${mono('Environment')}`, 'Keplerian, SGP4, RK4 orbits;', 'J2, drag, magnetic field,', 'ephemerides'])}
  ${box(336, 80, 266, 150, 'Satellite', ['inertia, geometry, disturbances', `actuators: ${mono('RW')}, ${mono('MTQ')}`, `sensors: ${mono('MTM')}, gyro, sun, star…`], { fill: t.accentSoft, stroke: t.accent })}
  ${box(652, 80, 214, 150, 'Estimator', ['EKF, MEKF, UKF, SRUKF', 'bias and disturbance states', 'orbit estimator (optional)'])}
  ${box(652, 286, 214, 136, 'Controller', ['PD, LQR, LP/QP allocation,', 'magnetorquer-only, B-dot,', 'plan-and-track'], { fill: t.okSoft, stroke: t.ok })}
  ${box(336, 286, 266, 136, 'Goal', [`${mono('Coordinate_Goal')}, nadir,`, 'three-axis, vector goals,', `${mono('GoalList')} schedules`], { fill: t.warmSoft, stroke: t.warm })}

  ${arrow('M276 155 L 332 155', 332, 155, R)}
  ${label(304, 145, 'B, sun, ρ')}
  ${arrow('M602 155 L 648 155', 648, 155, R)}
  ${label(625, 145, 'readings')}
  ${arrow('M759 230 L 759 282', 759, 282, D)}
  ${label(767, 262, 'estimate', 'start')}
  ${arrow('M602 354 L 648 354', 648, 354, R)}
  ${label(625, 344, 'target')}
  ${arrow('M866 354 Q 896 354 896 310 L 896 252 Q 896 240 884 240 L 469 240 L 469 234', 469, 234, U)}
  ${label(560, 256, 'actuator commands')}

  ${arrow('M884 150 L 930 150', 930, 150, R, t.accent)}
  ${box(934, 80, 322, 210, 'SimulationResults', ['state, estimate, control and', 'sensor histories', '', '→ ' + mono('ADCS.plot(...)'), '→ ' + mono('animate_orbit_godot(...)'), '→ ' + mono('results.save(...)')])}
  ${box(934, 306, 322, 116, 'Same loop, at scale', [mono('simulate_mc') + ': Monte Carlo', mono('simulate_remote') + ': hardware-in-', 'the-loop on a remote computer'])}
</svg>
`;
}

// Square project mark for the docs sidebar and favicon: Earth, an inclined
// orbit, and a spacecraft with its boresight, as in the banner.
function logo(t) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="160" height="160" viewBox="0 0 160 160" role="img" aria-label="Generalized ADCS logo">
  <ellipse cx="80" cy="84" rx="70" ry="26" transform="rotate(-24 80 84)" fill="none" stroke="${t.orbit}" stroke-width="3" stroke-dasharray="7 7"/>
  <circle cx="80" cy="84" r="34" fill="${t.accentSoft}" stroke="${t.accent}" stroke-width="3"/>
  <path d="M46 84 Q 80 70 114 84" fill="none" stroke="${t.accent}" stroke-opacity="0.5" stroke-width="2.5"/>
  <path d="M51 100 Q 80 89 109 100" fill="none" stroke="${t.accent}" stroke-opacity="0.4" stroke-width="2.5"/>
  <path d="M113 52 L 100 68" stroke="${t.accent}" stroke-width="2.5" stroke-dasharray="3 4"/>
  <g transform="translate(128 39) rotate(-36)">
    <rect x="-7" y="-7" width="14" height="14" rx="2" fill="${t.accent}"/>
    <rect x="-24" y="-4" width="13" height="8" rx="1.5" fill="none" stroke="${t.accent}" stroke-width="2.5"/>
    <rect x="11" y="-4" width="13" height="8" rx="1.5" fill="none" stroke="${t.accent}" stroke-width="2.5"/>
  </g>
</svg>
`;
}

for (const [name, t] of Object.entries(THEMES)) {
  fs.writeFileSync(path.join(OUT, `banner-${name}.svg`), banner(t));
  fs.writeFileSync(path.join(OUT, `architecture-${name}.svg`), diagram(t));
  fs.writeFileSync(path.join(OUT, `logo-${name}.svg`), logo(t));
}
console.log('wrote', fs.readdirSync(OUT).filter((f) => f.endsWith('.svg')).join(', '));
