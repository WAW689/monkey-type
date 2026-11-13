import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

// ---------------------------------------------------------------- 数据
const $ = (id) => document.getElementById(id);
const esc = (c) => c.replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
let R = null, TARGET = '', EXPS = {};
const match = (s) => {
  let n = 0;
  for (let i = 0; i < TARGET.length; i++) if (s[i] === TARGET[i]) n++;
  return n;
};

function buildExps() {
  const E = {
    random: {
      label: '① 纯随机', stepLabel: '尝试次数',
      desc: () => `每次在 ${R.vocab_size} 个字的键盘上随机敲 ${TARGET.length} 下，没有任何反馈。` +
        `搜索空间约 10<sup>${R.exp1.theory.log10_search_space.toFixed(1)}</sup>，即使每秒 10 亿次，也要约 ` +
        `10<sup>${R.exp1.theory.log10_universe_ages.toFixed(1)}</sup> 个宇宙年龄。` +
        `本次 ${R.exp1.attempts.toLocaleString()} 次尝试最多只对了 ${R.exp1.best_match} 个字。`,
      frames: R.exp1.history.map((h) => ({ step: h.attempt, text: h.sample, extra: `历史最好 ${h.best} 字` })),
      extraLabel: '记录',
    },
    weasel: {
      label: '② 黄鼠狼进化', stepLabel: '代数',
      desc: () => `累积选择：每代复制最优串 ${R.exp2.pop} 份，每个字以 ${R.exp2.mu} 的概率变异，` +
        `只保留最像目标的那一个。第 <b>${R.exp2.generations}</b> 代命中，只评估了 ${R.exp2.evaluations.toLocaleString()} 个串。` +
        `曲线中间的长平台是适应度平台：历史最优暂时不变，种群仍在变异，等一个低概率的正确突变。`,
      frames: R.exp2.history.map((h) => ({ step: h.gen, text: h.best })),
    },
  };
  for (const [mode, r] of Object.entries(R.exp3)) {
    E['rl_' + mode] = {
      label: `③ RL·${{ sparse: '稀疏', fluency: '通顺', target: '目标', mixed: '混合' }[mode]}`,
      stepLabel: '训练步', extraLabel: '贪心输出 / 多样性',
      desc: () => RL_DESC[mode] + (r.hit_step != null ? ` 第 <b>${r.hit_step}</b> 步首次打出目标句。` : ' 训练结束仍未打出目标句。'),
      frames: r.history.map((h) => ({ step: h.step, text: h.best, extra: `${h.greedy} / ${h.distinct2.toFixed(2)}` })),
    };
  }
  for (const e of Object.values(E)) e.frames.forEach((f) => (f.score = match(f.text)));
  return E;
}
const RL_DESC = {
  sparse: '稀疏奖励：只有一字不差才给奖励。猴子几乎永远拿 0 分，学不到任何东西，等同于随机猴子。',
  fluency: '只奖励"通顺"（三元语言模型打分）。猴子学会了刷分，反复敲高频片段，越来越"像话"，却离目标越来越远。这就是 reward hacking 和模式坍缩。',
  target: '奖励 = 与目标句逐位相同的字数。高于平均的句子被激励，低于平均的被惩罚（REINFORCE）。',
  mixed: '奖励 = 通顺度 + 0.6 × 正确字数。先学会"说人话"，但也可能被困在通顺的局部最优里（看多种子结果）。',
};
try {
  R = await (await fetch('./data/results.json')).json();
} catch (e) {
  $('err').style.display = 'flex';
  $('err').innerHTML = '读不到 data/results.json。<br>请先在项目根目录运行 <code>py run_all.py</code>，' +
    '再用 <code>py -m http.server 8765 -d web</code> 启动后打开 http://localhost:8765';
  throw e;
}
TARGET = R.target;
EXPS = buildExps();

// ---------------------------------------------------------------- 场景
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
$('scene').appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161c);
scene.fog = new THREE.Fog(0x14161c, 7, 16);
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
const HOME_POS = new THREE.Vector3(1.0, 2.9, 2.5);
const HOME_TARGET = new THREE.Vector3(0, 0.95, 0.1);
const INTRO_POS = new THREE.Vector3(-2.4, 2.2, 3.2);
const INTRO_TARGET = new THREE.Vector3(-0.8, 1.1, 0.1);
let introCamAnim = null;
camera.position.copy(HOME_POS);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(HOME_TARGET);
controls.enableDamping = true;
controls.enablePan = true;
controls.screenSpacePanning = true;
controls.minDistance = 1.1; controls.maxDistance = 10;
controls.minPolarAngle = 0.12;
controls.maxPolarAngle = Math.PI * 0.52;
controls.autoRotateSpeed = 2.0;

scene.add(new THREE.HemisphereLight(0xfff1dd, 0x202430, 0.7));
const sun = new THREE.DirectionalLight(0xfff0d8, 2.2);
sun.position.set(2.5, 5, 3);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3 });
scene.add(sun);
const lamp = new THREE.PointLight(0xffc47a, 6, 5, 2);
lamp.position.set(-0.9, 1.9, -0.5);
scene.add(lamp);

const mat = (color, o = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.75, ...o });
const mesh = (geo, m, parent, pos = [0, 0, 0], shadow = true) => {
  const x = new THREE.Mesh(geo, m);
  x.position.set(...pos);
  x.castShadow = shadow; x.receiveShadow = true;
  parent.add(x);
  return x;
};

// 地面与桌子
mesh(new THREE.CircleGeometry(7, 64).rotateX(-Math.PI / 2), mat(0x262a33), scene, [0, 0, 0], false);
const wood = mat(0x8a5a3b, { roughness: 0.6 });
mesh(new THREE.BoxGeometry(2.0, 0.08, 1.1), wood, scene, [0, 0.76, -0.2]);
for (const [x, z] of [[-0.9, -0.67], [0.9, -0.67], [-0.9, 0.27], [0.9, 0.27]])
  mesh(new THREE.BoxGeometry(0.07, 0.76, 0.07), wood, scene, [x, 0.38, z]);
// 台灯
const lampG = new THREE.Group(); scene.add(lampG); lampG.position.set(-0.8, 0.8, -0.55);
mesh(new THREE.CylinderGeometry(0.1, 0.12, 0.03, 24), mat(0x333333), lampG);
mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.9, 8), mat(0x333333), lampG, [0, 0.45, 0]);
mesh(new THREE.ConeGeometry(0.14, 0.18, 24, 1, true), mat(0x2f6f5e, { side: THREE.DoubleSide }), lampG, [0.03, 0.95, 0.03]);
// 香蕉（奖励）
const bananas = [];
for (let i = 0; i < 3; i++) {
  const curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(-0.12, 0, 0), new THREE.Vector3(0, 0.06, 0), new THREE.Vector3(0.12, 0.03, 0));
  const b = mesh(new THREE.TubeGeometry(curve, 12, 0.028, 8), mat(0xf2d14e, { roughness: 0.5 }), scene, [0.72, 0.83 + i * 0.03, -0.35 + i * 0.05]);
  b.rotation.y = i * 0.6;
  bananas.push(b);
}

// 桌上相框：图片比例自适应，最大高度 0.40、最大宽度 0.32
const frameGroup = new THREE.Group(); scene.add(frameGroup);
const frameWood = mat(0x2a1a10, { roughness: 0.55, metalness: 0.08 });
const frameMatBoard = mat(0xe8dfc9, { roughness: 0.95 });
const frameTex = new THREE.TextureLoader().load('./assets/dog-story.jpg', (tex) => {
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  const aspect = tex.image.width / tex.image.height;
  const maxH = 0.50, maxW = 0.38;
  let h = maxH, w = h * aspect;
  if (w > maxW) { w = maxW; h = w / aspect; }
  const border = 0.022, baseH = 0.022;
  const outerW = w + border * 2, outerH = h + border * 2;
  frameGroup.position.set(-0.68, 0.8, -0.01);
  frameGroup.rotation.y = 0.30;
  frameGroup.rotation.x = -0.04;

  // 底座
  mesh(new THREE.BoxGeometry(outerW * 0.82, baseH, 0.10), frameWood, frameGroup, [0, baseH / 2, 0.008]);
  // 背板 + 卡纸
  mesh(new THREE.BoxGeometry(outerW, outerH, 0.016), frameWood, frameGroup, [0, baseH + outerH / 2, -0.008]);
  mesh(new THREE.PlaneGeometry(w + 0.018, h + 0.018), frameMatBoard, frameGroup, [0, baseH + outerH / 2, 0.003], false);
  // 照片
  mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex }), frameGroup, [0, baseH + outerH / 2, 0.006], false);
  // 四周边框
  const barD = 0.026;
  mesh(new THREE.BoxGeometry(outerW, border, barD), frameWood, frameGroup, [0, baseH + outerH - border / 2, 0.002]);
  mesh(new THREE.BoxGeometry(outerW, border, barD), frameWood, frameGroup, [0, baseH + border / 2, 0.002]);
  mesh(new THREE.BoxGeometry(border, outerH, barD), frameWood, frameGroup, [-(outerW / 2 - border / 2), baseH + outerH / 2, 0.002]);
  mesh(new THREE.BoxGeometry(border, outerH, barD), frameWood, frameGroup, [outerW / 2 - border / 2, baseH + outerH / 2, 0.002]);
});

// 打字机
const tw = new THREE.Group(); scene.add(tw); tw.position.set(0, 0.8, -0.16);
const black = mat(0x1d1f24, { roughness: 0.35, metalness: 0.3 });
mesh(new THREE.BoxGeometry(0.9, 0.16, 0.55), black, tw, [0, 0.08, 0]);
mesh(new THREE.BoxGeometry(0.9, 0.12, 0.25).translate(0, 0.06, 0), black, tw, [0, 0.16, -0.15]);
const roller = mesh(new THREE.CylinderGeometry(0.06, 0.06, 1.0, 24).rotateZ(Math.PI / 2), mat(0x111111, { roughness: 0.3 }), tw, [0, 0.33, -0.2]);
for (const s of [-1, 1]) mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.05, 16).rotateZ(Math.PI / 2), mat(0xb8b8b8, { metalness: 0.8, roughness: 0.3 }), tw, [s * 0.53, 0.33, -0.2]);
const keys = [];
const keyMat = mat(0xe9e2cf, { roughness: 0.5 });
for (let row = 0; row < 4; row++)
  for (let col = 0; col < 10 - (row % 2); col++) {
    const k = mesh(new THREE.CylinderGeometry(0.026, 0.026, 0.03, 16), keyMat, tw,
      [-0.36 + col * 0.08 + (row % 2) * 0.04, 0.18 + row * 0.012, 0.22 - row * 0.075]);
    k.userData.y0 = k.position.y; k.userData.t = 0;
    keys.push(k);
  }
// 纸张（CanvasTexture 显示打出来的字）
const paperCanvas = document.createElement('canvas');
paperCanvas.width = 768; paperCanvas.height = 900;
const pctx = paperCanvas.getContext('2d');
const paperTex = new THREE.CanvasTexture(paperCanvas);
paperTex.colorSpace = THREE.SRGBColorSpace;
paperTex.anisotropy = 8;
const paper = mesh(new THREE.PlaneGeometry(0.62, 0.72), new THREE.MeshStandardMaterial({ map: paperTex, roughness: 0.9, side: THREE.DoubleSide }), tw, [0, 0.66, -0.24]);
paper.rotation.x = -0.18;

// 猴子（全部由基本几何体拼成）
const fur = mat(0x6b4226), skin = mat(0xd9a77a), dark = mat(0x1a1a1a, { roughness: 0.3 });
const monkey = new THREE.Group(); scene.add(monkey); monkey.position.set(0, 0, 0.62);
monkey.rotation.y = 0; // 面向 -Z（桌子与打字机），背对默认相机
// 凳子
mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.05, 24), wood, scene, [0, 0.5, 0.62]);
mesh(new THREE.CylinderGeometry(0.04, 0.06, 0.5, 12), wood, scene, [0, 0.25, 0.62]);
const body = new THREE.Group(); monkey.add(body); body.position.y = 0.53; body.rotation.x = -0.05;
mesh(new THREE.SphereGeometry(0.2, 24, 16).scale(1, 1.25, 0.85), fur, body, [0, 0.24, 0]);
mesh(new THREE.SphereGeometry(0.13, 20, 14).scale(1, 1.3, 0.5), skin, body, [0, 0.22, -0.1]);
const head = new THREE.Group(); body.add(head); head.position.set(0, 0.6, -0.02);
mesh(new THREE.SphereGeometry(0.17, 24, 18), fur, head);
mesh(new THREE.SphereGeometry(0.12, 20, 14).scale(1.15, 1, 0.6), skin, head, [0, -0.01, -0.12]);
mesh(new THREE.SphereGeometry(0.08, 16, 12).scale(1.3, 0.85, 0.9), skin, head, [0, -0.07, -0.16]);
for (const s of [-1, 1]) {
  mesh(new THREE.SphereGeometry(0.06, 16, 12).scale(1, 1, 0.4), fur, head, [s * 0.17, 0.02, 0]);
  mesh(new THREE.SphereGeometry(0.035, 12, 10).scale(1, 1, 0.4), skin, head, [s * 0.175, 0.02, -0.02]);
  mesh(new THREE.SphereGeometry(0.022, 12, 10), dark, head, [s * 0.05, 0.04, -0.2]);
  mesh(new THREE.SphereGeometry(0.008, 8, 6), mat(0xffffff, { emissive: 0xffffff }), head, [s * 0.05 + 0.008, 0.05, -0.22], false);
  mesh(new THREE.SphereGeometry(0.008, 8, 6), dark, head, [s * 0.025, -0.06, -0.235], false);
}
const mouth = mesh(new THREE.TorusGeometry(0.035, 0.007, 8, 20, Math.PI), dark, head, [0, -0.1, -0.225], false);
mouth.rotation.z = Math.PI; // 微笑
const brows = [-1, 1].map((s) => {
  const b = mesh(new THREE.BoxGeometry(0.05, 0.01, 0.01), dark, head, [s * 0.05, 0.09, -0.19], false);
  b.userData.s = s; return b;
});
// 手臂：肩 → 上臂 → 肘 → 前臂 → 手
const arms = [-1, 1].map((s) => {
  const shoulder = new THREE.Group(); body.add(shoulder); shoulder.position.set(s * 0.2, 0.4, -0.02);
  mesh(new THREE.CylinderGeometry(0.045, 0.04, 0.28, 12).translate(0, -0.14, 0), fur, shoulder);
  const elbow = new THREE.Group(); shoulder.add(elbow); elbow.position.y = -0.28;
  mesh(new THREE.CylinderGeometry(0.04, 0.035, 0.26, 12).translate(0, -0.13, 0), fur, elbow);
  mesh(new THREE.SphereGeometry(0.05, 14, 10).scale(1, 0.7, 1.2), skin, elbow, [0, -0.28, 0]);
  shoulder.userData = { s, t: 1 };
  return { shoulder, elbow };
});
for (const s of [-1, 1]) {
  const leg = new THREE.Group(); body.add(leg); leg.position.set(s * 0.1, 0.02, -0.05);
  mesh(new THREE.CylinderGeometry(0.06, 0.05, 0.3, 12).rotateX(Math.PI / 2).translate(0, 0, -0.15), fur, leg);
  mesh(new THREE.CylinderGeometry(0.045, 0.04, 0.4, 12).translate(0, -0.2, 0), fur, leg, [0, 0, -0.3]);
  mesh(new THREE.SphereGeometry(0.06, 14, 10).scale(1, 0.5, 1.6), skin, leg, [0, -0.42, -0.35]);
}
const tailCurve = new THREE.CatmullRomCurve3([[0, 0.1, 0.15], [0, 0.0, 0.4], [0.1, 0.15, 0.6], [0.25, 0.45, 0.55], [0.2, 0.6, 0.4]].map((p) => new THREE.Vector3(...p)));
const tail = mesh(new THREE.TubeGeometry(tailCurve, 40, 0.025, 8), fur, body);

// 彩纸（命中目标时）
const CONF = 300;
const confetti = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.03, 0.05), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }), CONF);
const confData = Array.from({ length: CONF }, () => ({ p: new THREE.Vector3(), v: new THREE.Vector3(), r: new THREE.Euler() }));
const palette = [0xf2c14e, 0x7bd389, 0xef6f6c, 0x6fa8dc, 0xffffff].map((c) => new THREE.Color(c));
for (let i = 0; i < CONF; i++) confetti.setColorAt(i, palette[i % palette.length]);
confetti.visible = false; scene.add(confetti);
let confT = -1;
function burst() {
  confT = 0; confetti.visible = true;
  for (const d of confData) {
    d.p.set((Math.random() - 0.5) * 0.4, 1.6, -0.2 + (Math.random() - 0.5) * 0.4);
    d.v.set((Math.random() - 0.5) * 2.5, 1.5 + Math.random() * 2.5, (Math.random() - 0.5) * 2.5);
    d.r.set(Math.random() * 6, Math.random() * 6, 0);
  }
}

// ---------------------------------------------------------------- 回放状态
let exp, expKey, fi = 0, ci = 0, playing = true, acc = 0, lines = [], mood = 0, moodT = 0, celebrate = 0, finished = false;
let trainingActive = false;
const tabs = $('tabs');
for (const [k, e] of Object.entries(EXPS)) {
  const b = document.createElement('button');
  b.textContent = e.label; b.dataset.k = k;
  b.onclick = () => select(k);
  tabs.appendChild(b);
}
$('tgt').textContent = TARGET;
$('targetInput').value = TARGET;

function select(k) {
  if (!EXPS[k]) return;
  expKey = k; exp = EXPS[k];
  [...tabs.children].forEach((b) => b.classList.toggle('on', b.dataset.k === k));
  $('desc').innerHTML = exp.desc();
  $('stepLabel').textContent = exp.stepLabel;
  $('extraRow').style.display = exp.extraLabel ? '' : 'none';
  $('extraLabel').textContent = exp.extraLabel || '';
  playing = true;
  setMood('🙂 准备中', 'var(--fg)');
  seekTo(0);
}
function applyResults(data) {
  R = data; TARGET = R.target;
  EXPS = buildExps();
  $('tgt').textContent = TARGET;
  $('targetInput').value = TARGET;
  select(expKey && EXPS[expKey] ? expKey : 'weasel');
}
function seekTo(i) {
  fi = Math.max(0, Math.min(exp.frames.length - 1, i));
  ci = 0; acc = 0; finished = false;
  lines = exp.frames.slice(Math.max(0, fi - 8), fi).map((f) => f.text);
  celebrate = 0; mood = 0;
  $('banner').classList.remove('show');
  $('play').textContent = playing ? '暂停' : '继续';
  updateHud();
}
function updateHud() {
  const f = exp.frames[fi];
  $('step').textContent = f.step.toLocaleString();
  const L = Math.max(1, TARGET.length);
  const typed = [...f.text].slice(0, ci);
  $('typed').style.fontSize = `${Math.max(13, Math.min(22, 320 / L))}px`;
  $('typed').style.letterSpacing = `${L > 13 ? 1 : 3}px`;
  $('typed').innerHTML = typed.map((c, i) => `<span class="${c === TARGET[i] ? 'ok' : 'no'}">${esc(c)}</span>`).join('') + (ci < L ? '<span class="no">▌</span>' : '');
  $('tline').textContent = '目标 ' + TARGET;
  $('score').textContent = `${match(typed.join(''))} / ${L}`;
  $('extra').textContent = f.extra || '';
  $('seek').value = Math.round((fi / Math.max(1, exp.frames.length - 1)) * 1000);
  drawChart();
  drawPaper(typed.join(''));
}

// 每打完一行：根据得分变化给出"激励"或"惩罚"
function finishLine() {
  const f = exp.frames[fi];
  lines.push(f.text); if (lines.length > 9) lines.shift();
  const prev = fi > 0 ? exp.frames[fi - 1].score : 0;
  const last = fi >= exp.frames.length - 1;
  if (f.text === TARGET) {
    mood = 2; celebrate = 1; burst();
    $('banner').textContent = TARGET + '！';
    $('banner').classList.add('show');
    setMood('🎉 打出来了！奖励一大串香蕉', 'var(--gold)');
    Sound.success();
    playing = false; finished = true; $('play').textContent = '重播';
    return;
  }
  if (f.score > prev) { mood = 1; setMood(`🍌 激励！多对了 ${f.score - prev} 个字`, 'var(--good)'); }
  else if (f.score < prev) { mood = -1; setMood(`⚡ 惩罚！少了 ${prev - f.score} 个字`, 'var(--bad)'); }
  else { mood = 0; setMood('🙂 没进步也没退步', 'var(--fg)'); }
  if (last) Sound.fail();
  else if (f.score > prev) Sound.up();
  else if (f.score < prev) Sound.down();
  moodT = 0;
  if (!last) { fi++; ci = 0; }
  else { playing = false; finished = true; $('play').textContent = '重播'; }
}
function setMood(t, c) { $('mood').textContent = t; $('mood').style.color = c; }

// ---------------------------------------------------------------- 绘制：纸张与曲线
function drawPaper(current) {
  const W = paperCanvas.width, H = paperCanvas.height;
  pctx.fillStyle = '#f4efe3'; pctx.fillRect(0, 0, W, H);
  pctx.strokeStyle = 'rgba(120,140,180,.25)'; pctx.lineWidth = 2;
  for (let y = 120; y < H; y = y + 78) { pctx.beginPath(); pctx.moveTo(30, y); pctx.lineTo(W - 30, y); pctx.stroke(); }
  pctx.font = '28px "Microsoft YaHei","PingFang SC",sans-serif';
  pctx.fillStyle = '#9a8f7a';
  pctx.fillText(`${exp.label}　${exp.stepLabel} ${exp.frames[fi].step.toLocaleString()}`, 40, 60);
  const all = [...lines, current];
  const show = all.slice(-10);
  const L = Math.max(1, TARGET.length);
  const cw = Math.min(52, (W - 80) / L);
  const fs = Math.min(46, cw * 0.92);
  pctx.font = `${fs}px "SimSun","Songti SC","Noto Serif CJK SC",serif`;
  show.forEach((s, li) => {
    const y = 110 + li * 78;
    [...s].forEach((c, i) => {
      pctx.fillStyle = s === TARGET ? '#b8860b' : c === TARGET[i] ? '#1e7a3a' : '#2a2a2a';
      pctx.globalAlpha = li === show.length - 1 ? 1 : 0.35 + 0.6 * (li / show.length);
      pctx.fillText(c, 40 + i * cw, y);
    });
  });
  pctx.globalAlpha = 1;
  pctx.strokeStyle = 'rgba(160,145,120,.28)'; pctx.lineWidth = 1;
  pctx.beginPath(); pctx.moveTo(40, 838); pctx.lineTo(W - 40, 838); pctx.stroke();
  pctx.font = '20px "Microsoft YaHei","PingFang SC",sans-serif';
  pctx.fillStyle = 'rgba(130,115,92,.55)';
  pctx.fillText('上海对外经贸大学 · 山狗', 40, 872);
  paperTex.needsUpdate = true;
}
const chart = $('chart'), cctx = chart.getContext('2d');
function drawChart() {
  const dpr = devicePixelRatio;
  const w = (chart.width = chart.clientWidth * dpr), h = (chart.height = chart.clientHeight * dpr);
  cctx.clearRect(0, 0, w, h);
  const F = exp.frames, n = F.length, pad = 6 * dpr;
  const L = Math.max(1, TARGET.length);
  const X = (i) => pad + (i / Math.max(1, n - 1)) * (w - 2 * pad), Y = (v) => h - pad - (v / L) * (h - 2 * pad);
  cctx.strokeStyle = '#2c303b'; cctx.lineWidth = dpr;
  for (const v of [0, L]) { cctx.beginPath(); cctx.moveTo(pad, Y(v)); cctx.lineTo(w - pad, Y(v)); cctx.stroke(); }
  cctx.strokeStyle = '#555a66'; cctx.lineWidth = 1.5 * dpr; cctx.beginPath();
  F.forEach((f, i) => (i ? cctx.lineTo(X(i), Y(f.score)) : cctx.moveTo(X(i), Y(f.score)))); cctx.stroke();
  cctx.strokeStyle = '#f2c14e'; cctx.lineWidth = 2 * dpr; cctx.beginPath();
  F.slice(0, fi + 1).forEach((f, i) => (i ? cctx.lineTo(X(i), Y(f.score)) : cctx.moveTo(X(i), Y(f.score)))); cctx.stroke();
  cctx.fillStyle = '#f2c14e'; cctx.beginPath(); cctx.arc(X(fi), Y(F[fi].score), 3.5 * dpr, 0, 7); cctx.fill();
  cctx.fillStyle = '#9a9ca3'; cctx.font = `${10 * dpr}px sans-serif`;
  cctx.fillText(String(L), pad + 2, Y(L) + 11 * dpr); cctx.fillText('正确字数', w - 52 * dpr, h - 10 * dpr);
}

// ---------------------------------------------------------------- 控件
$('play').onclick = () => {
  if (!playing && finished) { seekTo(0); playing = true; $('play').textContent = '暂停'; return; }
  playing = !playing; $('play').textContent = playing ? '暂停' : '继续';
};
const speed = () => +$('speed').value;
const speedLabel = () => { const s = speed(); $('speedv').textContent = s > 40 ? '飞快' : `${s} 字/秒`; };
$('speed').oninput = speedLabel; speedLabel();
$('seek').oninput = (e) => seekTo(Math.round((e.target.value / 1000) * (exp.frames.length - 1)));

// 自由视角：按钮绕目标点旋转；鼠标拖拽/滚轮/右键由 OrbitControls 接管
const sph = new THREE.Spherical();
function orbitBy(dTheta, dPhi = 0) {
  sph.setFromVector3(camera.position.clone().sub(controls.target));
  sph.theta += dTheta;
  sph.phi = THREE.MathUtils.clamp(sph.phi + dPhi, controls.minPolarAngle, controls.maxPolarAngle);
  const damp = controls.enableDamping;
  controls.enableDamping = false; // 先清掉拖拽残留的阻尼量，避免按钮转动被带偏
  controls.update();
  camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(sph));
  controls.update();
  controls.enableDamping = damp;
}
function resetView() {
  if (controls.autoRotate) {
    controls.autoRotate = false;
    $('vAuto').classList.remove('on');
    $('vAuto').textContent = '自动';
  }
  const damp = controls.enableDamping;
  controls.enableDamping = false; // 清掉拖拽残留的阻尼速度
  controls.update();
  camera.position.copy(HOME_POS);
  controls.target.copy(HOME_TARGET);
  controls.update();
  controls.enableDamping = damp;
}
$('vLeft').onclick = () => orbitBy(-Math.PI / 12);
$('vRight').onclick = () => orbitBy(Math.PI / 12);
$('vReset').onclick = resetView;
$('vAuto').onclick = () => {
  controls.autoRotate = !controls.autoRotate;
  $('vAuto').classList.toggle('on', controls.autoRotate);
  $('vAuto').textContent = controls.autoRotate ? '停止' : '自动';
};

// ---------------------------------------------------------------- 音效（Web Audio 合成，无需音频文件）
let soundOn = false;
const Sound = (() => {
  let ctx = null, master = null, noiseBuf = null;
  let volume = 0.3;
  let lastClick = 0, lastFeedback = 0;
  const stats = { click: 0, up: 0, down: 0, success: 0, fail: 0, ping: 0 };

  function ensure() {
    if (ctx) {
      if (ctx.state !== 'running') ctx.resume().catch(() => {});
      return true;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return false;
    ctx = new AC();
    master = ctx.createGain();
    master.gain.value = volume;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -14; comp.knee.value = 12; comp.ratio.value = 4;
    // 总线上加低通，去掉高频毛刺，整体更柔和
    const tone = ctx.createBiquadFilter();
    tone.type = 'lowpass'; tone.frequency.value = 2600; tone.Q.value = 0.5;
    master.connect(tone); tone.connect(comp); comp.connect(ctx.destination);
    const n = Math.floor(ctx.sampleRate * 0.08);
    noiseBuf = ctx.createBuffer(1, n, ctx.sampleRate);
    const d = noiseBuf.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
    if (ctx.state !== 'running') ctx.resume().catch(() => {});
    return true;
  }
  const ready = () => soundOn && ensure() && ctx.state === 'running';
  function setVolume(v) {
    volume = Math.max(0, Math.min(1, v));
    if (master) master.gain.value = volume;
  }

  // 单个音符：指数包络，听起来像"叮"
  function blip(freq, dur, { type = 'sine', gain = 0.1, delay = 0, slide = null } = {}) {
    const t = ctx.currentTime + delay;
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, t);
    if (slide) o.frequency.exponentialRampToValueAtTime(slide, t + dur);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g); g.connect(master);
    o.start(t); o.stop(t + dur + 0.03);
  }

  // 键盘敲击：低通噪声（木质轻响）+ 低频短音，去掉刺耳的频段
  function click() {
    if (!ready()) return;
    const n = performance.now();
    if (n - lastClick < 30) return; // 高速播放时限流，避免变成机关枪
    lastClick = n;
    stats.click++;
    const t = ctx.currentTime;
    const src = ctx.createBufferSource(); src.buffer = noiseBuf;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 700 + Math.random() * 300;
    lp.Q.value = 0.7;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.22 + Math.random() * 0.06, t + 0.003);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.045);
    src.connect(lp); lp.connect(g); g.connect(master);
    src.start(t); src.stop(t + 0.06);
    blip(120 + Math.random() * 30, 0.05, { gain: 0.06, slide: 85 });
  }
  function up() {
    if (!ready()) return;
    const n = performance.now();
    if (n - lastFeedback < 160) return;
    lastFeedback = n;
    stats.up++;
    blip(523.25, 0.22, { gain: 0.09 });
    blip(659.25, 0.28, { gain: 0.07, delay: 0.08 });
  }
  function down() {
    if (!ready()) return;
    const n = performance.now();
    if (n - lastFeedback < 160) return;
    lastFeedback = n;
    stats.down++;
    blip(246.94, 0.22, { gain: 0.07, slide: 196 });
    blip(196, 0.3, { gain: 0.06, delay: 0.08, slide: 164.81 });
  }
  function success() {
    if (!ready()) return;
    lastFeedback = performance.now();
    stats.success++;
    [261.63, 329.63, 392, 523.25].forEach((f, i) =>
      blip(f, 0.45, { type: 'triangle', gain: 0.11, delay: i * 0.1 }));
    blip(523.25, 0.6, { type: 'sine', gain: 0.07, delay: 0.4 });
  }
  function fail() {
    if (!ready()) return;
    const n = performance.now();
    if (n - lastFeedback < 400) return;
    lastFeedback = n;
    stats.fail++;
    blip(261.63, 0.25, { gain: 0.08, slide: 246.94 });
    blip(196, 0.35, { gain: 0.07, delay: 0.14, slide: 174.61 });
    blip(130.81, 0.5, { gain: 0.05, delay: 0.28 });
  }
  // 开声音时的试听：两声柔和的“叮”
  function ping() {
    if (!ready()) return;
    stats.ping++;
    blip(440, 0.18, { gain: 0.12 });
    blip(523.25, 0.26, { gain: 0.1, delay: 0.12 });
  }
  return {
    ensure, click, up, down, success, fail, ping, setVolume,
    get state() { return ctx ? ctx.state : 'none'; },
    get volume() { return volume; },
    get stats() { return { ...stats }; }
  };
})();

const soundBtn = $('sound');
function setSound(on, test = true) {
  soundOn = on;
  soundBtn.textContent = on ? '🔊 音效开' : '🔇 音效关';
  soundBtn.classList.toggle('on', on);
  if (on) {
    Sound.ensure();
    if (test) enableSoundTest();
  }
}
function enableSoundTest(attempt = 0) {
  Sound.ensure();
  if (Sound.state === 'running') { Sound.ping(); return; }
  if (attempt < 5) setTimeout(() => enableSoundTest(attempt + 1), 120);
}
soundBtn.onclick = () => setSound(!soundOn);
// 每次打开都从静音开始：避免“显示音效开、但浏览器还没解锁”的假开启状态
setSound(false, false);
addEventListener('pointerdown', () => { if (soundOn) Sound.ensure(); }, { passive: true });
soundBtn.title = '开关音效（打开时会试听两声）';

const vol = $('vol'), volv = $('volv');
const volLabel = () => { volv.textContent = vol.value; };
vol.oninput = () => { Sound.setVolume(+vol.value / 100); volLabel(); };
Sound.setVolume(+vol.value / 100);
volLabel();

// ---------------------------------------------------------------- 动画
let hand = 0;
function press() {
  ci++;
  hand = 1 - hand;
  arms[hand].shoulder.userData.t = 0;
  const k = keys[(Math.random() * keys.length) | 0]; k.userData.t = 0;
  roller.rotation.x += 0.15;
  Sound.click();
}
const clock = new THREE.Clock();
function tick() {
  const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
  if (playing) {
    // 速度很高时，每帧可以打完多行
    const cps = speed() > 40 ? (speed() - 40) * 40 + 60 : speed();
    acc += dt * cps;
    let guard = 0;
    while (acc >= 1 && playing && guard++ < 500) {
      acc -= 1;
      if (ci < TARGET.length) press(); else finishLine();
    }
    updateHud();
  }
  moodT += dt;

  // 手臂：按键时下落再抬起；欢庆时高举
  for (const { shoulder, elbow } of arms) {
    const u = shoulder.userData; u.t = Math.min(1, u.t + dt * 10);
    const hit = Math.sin(u.t * Math.PI);
    if (celebrate) {
      shoulder.rotation.set(Math.PI * 0.95 + Math.sin(t * 10 + u.s) * 0.15, 0, -u.s * 0.35);
      elbow.rotation.x = 0.2;
    } else {
      // 面朝 -Z：手臂向前伸到打字机键盘上，按键时下压
      shoulder.rotation.set(1.62 - hit * 0.04, 0, u.s * 0.12);
      elbow.rotation.x = 0.12 - hit * 0.04;
    }
  }
  for (const k of keys) { k.userData.t = Math.min(1, k.userData.t + dt * 12); k.position.y = k.userData.y0 - Math.sin(k.userData.t * Math.PI) * 0.012; }

  // 表情与身体：激励 → 点头跳一下；惩罚 → 摇头皱眉
  const fade = Math.max(0, 1 - moodT * 1.6);
  const m = celebrate ? 2 : mood;
  head.rotation.set(0, 0, 0);
  body.position.y = 0.53;
  if (m === 2) { body.position.y = 0.53 + Math.abs(Math.sin(t * 7)) * 0.12; head.rotation.z = Math.sin(t * 7) * 0.15; }
  else if (m === 1) { body.position.y = 0.53 + Math.sin(moodT * 12) * 0.04 * fade; head.rotation.x = -Math.sin(moodT * 14) * 0.15 * fade; }
  else if (m === -1) { head.rotation.y = Math.sin(moodT * 22) * 0.3 * fade; }
  head.rotation.x += Math.sin(t * 2) * 0.03 + (playing && !celebrate ? -0.25 : 0); // 低头看键盘
  mouth.rotation.z = m >= 1 || (m === 0) ? Math.PI : 0;
  mouth.position.y = m === -1 ? -0.115 : -0.1;
  for (const b of brows) b.rotation.z = (m === -1 ? -0.4 : m >= 1 ? 0.2 : 0) * b.userData.s;
  tail.rotation.z = Math.sin(t * 1.5) * 0.15 + (m >= 1 ? Math.sin(t * 12) * 0.2 : 0);
  bananas.forEach((b, i) => { b.visible = m >= 1 || celebrate; b.position.y = 0.83 + i * 0.03 + (celebrate ? Math.abs(Math.sin(t * 5 + i)) * 0.1 : 0); });
  lamp.intensity = m === -1 && fade > 0 ? 6 + Math.sin(moodT * 50) * 4 * fade : 6;

  if (confT >= 0) {
    confT += dt;
    const o = new THREE.Object3D();
    confData.forEach((d, i) => {
      d.v.y -= 3.5 * dt; d.v.multiplyScalar(0.99); d.p.addScaledVector(d.v, dt);
      if (d.p.y < 0.02) { d.p.y = 0.02; d.v.set(0, 0, 0); }
      d.r.x += dt * 6; d.r.y += dt * 4;
      o.position.copy(d.p); o.rotation.copy(d.r); o.updateMatrix(); confetti.setMatrixAt(i, o.matrix);
    });
    confetti.instanceMatrix.needsUpdate = true;
    if (confT > 8) { confT = -1; confetti.visible = false; }
  }

  if (introCamAnim) {
    const a = introCamAnim;
    const k = Math.min(1, (performance.now() - a.start) / (a.dur * 1000));
    const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    camera.position.lerpVectors(a.fromP, a.toP, e);
    controls.target.lerpVectors(a.fromT, a.toT, e);
    controls.update();
    if (k >= 1) { introCamAnim = null; if (a.done) a.done(); }
  }
  if (!trainingActive) {
    controls.update(dt);
    renderer.render(scene, camera);
  }
  requestAnimationFrame(tick);
}
function resize() {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  // 宽屏时把画面往右挪，避免被左侧面板挡住
  camera.setViewOffset(innerWidth, innerHeight, innerWidth > 900 ? -190 : 0, 0, innerWidth, innerHeight);
}
addEventListener('resize', resize); resize();
select('weasel');
tick();

// ---------------------------------------------------------------- 实时训练
let corpusPromise = null;
function getCorpus() {
  if (!corpusPromise) {
    corpusPromise = fetch('./data/corpus.txt')
      .then((r) => {
        if (!r.ok) throw new Error('corpus.txt 读取失败（HTTP ' + r.status + '）');
        return r.text();
      })
      .then((txt) => txt.split(/\r?\n/).map((s) => s.trim()).filter((s) => s && !s.startsWith('#')))
      .catch((err) => { corpusPromise = null; throw err; });
  }
  return corpusPromise;
}
let training = false;
const targetInput = $('targetInput'), targetBtn = $('targetTrain');
function validateTarget(raw) {
  const s = (raw || '').trim();
  const chars = [...s];
  if (chars.length < 4 || chars.length > 20) return '目标需要 4–20 个字';
  if (chars.some((c) => /\s/.test(c))) return '目标里不能有空格或换行';
  const bad = chars.find((c) => !/[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\x21-\x7e]/.test(c));
  if (bad) return `目标里有不支持的字符：${bad}`;
  return '';
}
function trainingFailed(message) {
  training = false;
  trainingActive = false;
  $('train').disabled = false;
  $('train').textContent = '🎲 再试一次';
  targetInput.disabled = false;
  targetBtn.disabled = false;
  targetBtn.textContent = '训练此目标';
  $('trainmsg').textContent = message;
}
function startTraining(targetOverride) {
  if (training) return;
  const target = (targetOverride || TARGET || '').trim();
  const err = validateTarget(target);
  if (err) { $('trainmsg').textContent = err; targetInput.focus(); return; }
  training = true;
  trainingActive = true;
  playing = false;
  $('play').textContent = '继续';
  const btn = $('train'), msg = $('trainmsg');
  btn.disabled = true; btn.textContent = '训练中…';
  targetInput.disabled = true; targetBtn.disabled = true; targetBtn.textContent = '训练中…';
  msg.textContent = '加载语料…';
  const t0 = performance.now();
  let seed = 0;
  try {
    const buf = new Uint32Array(1);
    crypto.getRandomValues(buf);
    seed = buf[0] >>> 0;
  } catch (e) {
    seed = (Math.random() * 4294967296) >>> 0;
  }

  getCorpus().then((corpus) => {
    msg.textContent = '训练中 · ① 随机';
    const worker = new Worker('./train.js');
    const done = (results) => {
      try {
        const secs = ((performance.now() - t0) / 1000).toFixed(1);
        trainingActive = false;
        applyResults(results);
        msg.textContent = `实时训练 · 种子 ${seed} · 用时 ${secs}s`;
        btn.disabled = false; btn.textContent = '🎲 再跑一次';
        targetInput.disabled = false;
        targetBtn.disabled = false;
        targetBtn.textContent = '训练此目标';
        training = false;
      } catch (err) {
        trainingFailed('应用结果失败：' + err.message);
      } finally {
        worker.terminate();
      }
    };
    worker.onmessage = (ev) => {
      const d = ev.data;
      if (d.type === 'progress') msg.textContent = `训练中 · ${d.label} ${d.pct}%`;
      else if (d.type === 'done') done(d.results);
      else if (d.type === 'error') { worker.terminate(); trainingFailed('训练出错：' + d.message); }
    };
    worker.onerror = (err) => { worker.terminate(); trainingFailed('Worker 出错：' + (err.message || err)); };
    worker.postMessage({ seed, corpus, target });
  }).catch((err) => trainingFailed('读不到 data/corpus.txt：' + err.message));
}
$('train').onclick = () => {
  if (training) return;
  // 如果输入框里改过目标，大按钮也按新目标训练，避免“点了没反应”
  const s = targetInput.value.trim();
  if (s && s !== TARGET) { applyCustomTarget(); return; }
  startTraining();
};
function applyCustomTarget() {
  if (training) return;
  const s = targetInput.value;
  const err = validateTarget(s);
  if (err) { $('trainmsg').textContent = err; targetInput.focus(); return; }
  startTraining(s.trim());
}
targetBtn.onclick = applyCustomTarget;
targetInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); applyCustomTarget(); }
});
getCorpus().catch(() => {}); // 提前预取，点按钮时不用等

// ---------------------------------------------------------------- 背景浮层
const INTRO_LINES = [
  '如果让一只猴子在打字机上随机敲键，只要时间足够长，它几乎必然会敲出莎士比亚的作品。',
  '这就是<em>无限猴子定理</em>：一个几乎不可能的事件，在无限次独立尝试下，概率趋近于 1。',
  '但它只回答“会不会”，不回答“要多久”。以默认的 373 个字符、13 个字为例，期望尝试次数约 10<sup>33.4</sup>。',
  '即使每秒尝试 10 亿次，也要约 620 万个宇宙年龄。',
  '所以真正的题目不是“等”，而是“怎样让它发生”。',
  '1986 年，Dawkins 用<em>黄鼠狼程序</em>演示了累积选择：保留已经对了一点点的中间结果，让指数搜索变成爬山。',
  '再进一步，<em>强化学习</em>让猴子从奖励里自己学。目标、通顺、稀疏、混合——四种奖励，塑造出四种不同的行为。',
  '这个项目把这三步放在同一只中文猴子身上。随机不可能，选择让它可能，学习让它高效；而奖励怎么设计，决定它到底学到什么。',
];
const introEl = $('intro'), introStage = $('introStage'), introLinesEl = $('introLines');
const introDotsEl = $('introDots'), introHint = $('introHint');
const introLineEls = [], introDotEls = [];
let introOpen = false, introIdx = -1, introTimer = 0, introDone = false, introWasPlaying = false;
let introCamSaved = null;
const INTRO_KEY = 'monkey-intro-seen';

function moveIntroCam(pos, target, dur, done) {
  introCamAnim = {
    fromP: camera.position.clone(), toP: pos.clone(),
    fromT: controls.target.clone(), toT: target.clone(),
    start: performance.now(), dur, done,
  };
}

function initIntro() {
  INTRO_LINES.forEach((html, i) => {
    const p = document.createElement('p');
    p.className = 'bg-line future';
    p.innerHTML = html;
    introLinesEl.appendChild(p);
    introLineEls.push(p);
    const d = document.createElement('span');
    introDotsEl.appendChild(d);
    introDotEls.push(d);
  });
  introEl.addEventListener('click', (e) => {
    if (!introOpen || e.target.closest('button')) return;
    if (introDone) closeIntro(); else advanceIntro();
  });
  $('about').onclick = openIntro;
  $('introClose').onclick = closeIntro;
  $('introDone').onclick = closeIntro;
  $('introReplay').onclick = resetIntro;
  addEventListener('keydown', (e) => {
    if (!introOpen) return;
    if (e.key === 'Escape') { e.preventDefault(); closeIntro(); }
    else if (e.key === ' ' || e.key === 'ArrowRight') {
      e.preventDefault();
      if (introDone) closeIntro(); else advanceIntro();
    }
  });
}
function markIntroSeen() { try { localStorage.setItem(INTRO_KEY, '1'); } catch (e) {} }
function resetIntro() {
  clearTimeout(introTimer);
  introIdx = -1; introDone = false;
  introLineEls.forEach((p) => { p.className = 'bg-line future'; });
  introDotEls.forEach((d) => { d.className = ''; });
  introStage.scrollTop = 0;
  introHint.textContent = '点击画面继续 · Esc 关闭';
  introTimer = setTimeout(advanceIntro, 500);
}
function showIntroLine(i) {
  const p = introLineEls[i];
  p.classList.remove('future');
  p.classList.add('active');
  introDotEls[i].classList.add('on');
  const target = p.offsetTop - introStage.clientHeight / 2 + p.offsetHeight / 2;
  introStage.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
}
function advanceIntro() {
  if (!introOpen || introDone) return;
  clearTimeout(introTimer);
  if (introIdx === introLineEls.length - 1) {
    introDone = true;
    introHint.textContent = '读完了 · 点击画面或「进入演示」';
    return;
  }
  if (introIdx >= 0) {
    const prev = introLineEls[introIdx];
    prev.classList.remove('active'); prev.classList.add('past');
    introDotEls[introIdx].classList.remove('on'); introDotEls[introIdx].classList.add('done');
  }
  introIdx++;
  showIntroLine(introIdx);
  const dwell = Math.min(5200, Math.max(2600, 1100 + introLineEls[introIdx].textContent.length * 42));
  introTimer = setTimeout(advanceIntro, dwell);
}
function openIntro() {
  if (introOpen) return;
  introOpen = true;
  introCamSaved = { pos: camera.position.clone(), target: controls.target.clone() };
  introWasPlaying = playing;
  if (playing) { playing = false; $('play').textContent = '继续'; }
  // 让开场保持干净：收起庆祝/彩纸/横幅
  celebrate = 0; mood = 0; moodT = 0;
  confT = -1; confetti.visible = false;
  $('banner').classList.remove('show');
  document.body.classList.add('intro-on');
  controls.enabled = false;
  moveIntroCam(INTRO_POS, INTRO_TARGET, 1.0);
  introEl.classList.add('open');
  introEl.setAttribute('aria-hidden', 'false');
  resetIntro();
  setTimeout(() => $('introClose').focus({ preventScroll: true }), 60);
}
function closeIntro() {
  if (!introOpen) return;
  introOpen = false;
  clearTimeout(introTimer);
  introEl.classList.remove('open');
  introEl.setAttribute('aria-hidden', 'true');
  markIntroSeen();
  const saved = introCamSaved;
  introCamSaved = null;
  document.body.classList.remove('intro-on');
  if (saved) {
    controls.enabled = false;
    moveIntroCam(saved.pos, saved.target, 0.9, () => { controls.enabled = true; });
  } else {
    controls.enabled = true;
  }
  if (introWasPlaying && !finished) { playing = true; $('play').textContent = '暂停'; }
  introWasPlaying = false;
  $('about').focus({ preventScroll: true });
}
initIntro();
let introSeen = false;
try { introSeen = localStorage.getItem(INTRO_KEY) === '1'; } catch (e) {}
if (!introSeen) {
  setTimeout(() => {
    let seenNow = false;
    try { seenNow = localStorage.getItem(INTRO_KEY) === '1'; } catch (e) {}
    if (!introOpen && !training && !seenNow) openIntro();
  }, 1000);
}

window.__monkey = {
  select, seekTo, orbitBy, resetView, camera, controls, train: startTraining,
  intro: { open: openIntro, close: closeIntro, get isOpen() { return introOpen; }, get index() { return introIdx; } },
  get summary() {
    return {
      target: TARGET,
      seed: R.seed, gens: R.exp2.generations, evals: R.exp2.evaluations,
      best: R.exp1.best_match,
      hits: Object.fromEntries(Object.entries(R.exp3).map(([k, v]) => [k, v.hit_step])),
    };
  },
  sound: {
    get on() { return soundOn; },
    get state() { return Sound.state; },
    get stats() { return Sound.stats; },
    toggle: () => setSound(!soundOn)
  },
  get state() { return { expKey, fi, ci, playing }; }
};
