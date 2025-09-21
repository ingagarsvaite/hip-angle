import { PoseLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/+esm';

// ===== UI =====
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

const perm = document.getElementById('perm');
const startCamBtn = document.getElementById('startCam');

const dot = document.getElementById('dot');
const label = document.getElementById('label');
const avgVal = document.getElementById('avgVal');
const abdLVal = document.getElementById('abdL');
const abdRVal = document.getElementById('abdR');
const warn = document.getElementById('warn');

const startBtn = document.getElementById('start');
const dlBtn = document.getElementById('dl');
const bar = document.getElementById('bar');
const dbg = document.getElementById('dbg');

// ===== Paciento kodas (1–10 skaitmenų) – kaip manual app =====
let patientCode = null;
function askPatientCode(){
  let ok = false;
  while(!ok){
    const val = prompt('Įveskite paciento/tyrimo kodą (iki 10 skaitmenų):', '');
    if (val === null) return false; // atšaukė
    if (/^\d{1,10}$/.test(val)) { patientCode = val; ok = true; }
    else alert('Kodas turi būti 1–10 skaitmenų.');
  }
  return true;
}

// ===== Įrašymo parametrai =====
const RECORD_MS = 2000;                 // 2 s
const SAMPLE_MS = 10;                   // 10 ms
const RECORD_SAMPLES = RECORD_MS / SAMPLE_MS; // 200 ėminių tiksliai

let collectedData = [];
let isCollecting = false;
let sampleIdx = 0;     // 0,1,2,... → time_s = (idx+1)*0.01
let sampler = null;

// ===== MediaPipe / landmarks =====
let pose;
const LM = {
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26
};

// ===== Vektoriai / kampai – identiška formulė kaip manual app =====
const unit = (v)=>{ const n=Math.hypot(v.x,v.y); return n>1e-6?{x:v.x/n,y:v.y/n}:{x:0,y:0}; };
const sub  = (a,b)=>({x:a.x-b.x, y:a.y-b.y});
const dotp = (a,b)=>a.x*b.x + a.y*b.y;
const ang  = (a,b)=>{ const c=Math.max(-1,Math.min(1, dotp(unit(a),unit(b)))); return Math.acos(c)*180/Math.PI; };

// Kūno vidurio ašis: S_mid → H_mid (pečių vidurio taškas į klubų vidurio tašką), nukreipta žemyn
function bodyMidlineFromLandmarks(L){
  const S_mid = { x:(L.leftShoulder.x + L.rightShoulder.x)/2, y:(L.leftShoulder.y + L.rightShoulder.y)/2 };
  const H_mid = { x:(L.leftHip.x + L.rightHip.x)/2,           y:(L.leftHip.y + L.rightHip.y)/2 };
  let midDown = unit(sub(H_mid, S_mid));
  if (midDown.y < 0) midDown = { x:-midDown.x, y:-midDown.y };
  return { S_mid, H_mid, midDown };
}

// Abdukcija = kampas tarp HIP→KNEE ir midDown (ta pati formulė)
function abductionPerHip2D(HIP, KNEE, midDown){
  return ang(sub(KNEE, HIP), midDown);
}

// ===== Spalviniai lygiai – tokie patys =====
const SAFE = { greenMin:30, greenMax:45, yellowMax:60 };
const colAbd = (a)=> a>=SAFE.greenMin && a<=SAFE.greenMax ? '#34a853'
                    : a>SAFE.greenMax && a<=SAFE.yellowMax ? '#f9ab00'
                    : '#ea4335';

// ===== Telefonos palinkimas (tilt) – kaip manual app =====
let tiltDeg = null;    // laipsniais
let tiltOK  = null;    // |tilt| <= 5°
let sensorsEnabled = false;

function updateTiltWarn(){
  if (tiltDeg == null) return;
  if (Math.abs(tiltDeg) > 5){
    warn.textContent = `⚠️ Telefonas pakreiptas ${tiltDeg.toFixed(1)}° (>5°). Ištiesinkite įrenginį.`;
  }
}

function onDeviceOrientation(e){
  const portrait = window.innerHeight >= window.innerWidth;
  const primaryTilt = portrait ? (e.gamma ?? 0) : (e.beta ?? 0); // apytiksliai °
  tiltDeg = Number(primaryTilt) || 0;
  tiltOK  = Math.abs(tiltDeg) <= 5;
  updateTiltWarn();
}

async function enableSensors(){
  try{
    // iOS: leidimas PRIVALO būti paprašytas vartotojo gesto metu
    if (typeof DeviceOrientationEvent !== 'undefined' &&
        typeof DeviceOrientationEvent.requestPermission === 'function'){
      const perm = await DeviceOrientationEvent.requestPermission();
      if (perm !== 'granted') throw new Error('Leidimas nesuteiktas');
    }
    window.addEventListener('deviceorientation', onDeviceOrientation, true);
    sensorsEnabled = true;
  }catch(e){
    console.warn('Nepavyko įjungti tilt jutiklio:', e);
  }
}

// ===== Kamera =====
async function initCamera(){
  // Paciento kodas
  if (!patientCode){
    const cont = askPatientCode();
    if (!cont) return;
  }
  // Tilt jutiklis (leidimas ant vartotojo gesto)
  await enableSensors();

  try{
    // Kamera
    let constraints = { video:{facingMode:{ideal:'environment'}, width:{ideal:1280}, height:{ideal:720}}, audio:false };
    let stream = await navigator.mediaDevices.getUserMedia(constraints).catch(()=>null);
    if (!stream){ stream = await navigator.mediaDevices.getUserMedia({video:true, audio:false}); }

    video.setAttribute('playsinline',''); video.setAttribute('muted',''); video.setAttribute('autoplay','');
    video.srcObject = stream;

    await new Promise(res => { if (video.readyState>=1) res(); else video.onloadedmetadata=res; });
    await video.play();

    // UI
    perm.style.display = 'none';
    label.textContent = 'Kamera aktyvi';
    resizeCanvas();

    // MediaPipe
    if (!pose) await initPose();

    startBtn.disabled = false;
    requestAnimationFrame(loop);
  } catch(err){
    console.error('Kameros klaida:', err);
    perm.style.display = 'block';
    label.textContent = 'Klaida: nepavyko pasiekti kameros';
    warn.textContent = 'Patikrink naršyklės leidimus (naršyklėje „Allow“ arba sistemoje Camera: Allow).';
  }
}

function resizeCanvas(){
  const w = video.clientWidth, h = video.clientHeight;
  if (!w || !h) return;
  canvas.width = w; canvas.height = h;
}
window.addEventListener('resize', resizeCanvas);

// ===== MediaPipe init =====
async function initPose(){
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
  );
  pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions:{
      modelAssetPath:'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
      delegate:'GPU'
    },
    runningMode:'VIDEO',
    numPoses:1,
    minPoseDetectionConfidence:0.5,
    minPosePresenceConfidence:0.5,
    minTrackingConfidence:0.5
  });
}
const visOK = (p)=> (p.visibility ?? 1) >= 0.6;

// ===== Būsena sampleriui =====
let latest = { ok:false, angles:null, midline:null, lms:null };

// ===== Overlay (pečiai/klubai/keliai + šlaunys + S_mid↔H_mid) =====
function drawOverlay(L, angles, midline){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!L) return;

  const toPx = (p)=>({ x:p.x*canvas.width, y:p.y*canvas.height });

  const LS = toPx(L.leftShoulder),  RS = toPx(L.rightShoulder);
  const LH = toPx(L.leftHip),       RH = toPx(L.rightHip);
  const LK = toPx(L.leftKnee),      RK = toPx(L.rightKnee);

  const Spt = toPx(midline.S_mid);
  const Hpt = toPx(midline.H_mid);

  // midline S_mid → H_mid
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(Spt.x, Spt.y); ctx.lineTo(Hpt.x, Hpt.y); ctx.stroke();

  // HIP→KNEE
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 5; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(LH.x,LH.y); ctx.lineTo(LK.x,LK.y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(RH.x,RH.y); ctx.lineTo(RK.x,RK.y); ctx.stroke();

  // žymos
  ctx.fillStyle = '#ffffff';
  for (const p of [LS, RS, LH, RH, LK, RK]){
    ctx.beginPath(); ctx.arc(p.x,p.y,4,0,Math.PI*2); ctx.fill();
  }

  // kampų etiketės
  const colorL = colAbd(angles.abdL);
  const colorR = colAbd(angles.abdR);
  ctx.font = '14px system-ui, -apple-system, Segoe UI, Roboto';
  ctx.textBaseline = 'bottom';

  function labelAt(pt, text, col){
    const pad = 4, offY = -8;
    const m = ctx.measureText(text);
    const w = m.width + pad*2, h = 18;
    const x = Math.min(Math.max(0, pt.x - w/2), canvas.width - w);
    const y = Math.min(Math.max(h, pt.y + offY), canvas.height);
    ctx.fillStyle = 'rgba(0,0,0,0.55)'; ctx.fillRect(x, y-h, w, h);
    ctx.fillStyle = col; ctx.fillText(text, x + pad, y - 4);
  }
  if (isFinite(angles.abdL)) labelAt(LH, `${angles.abdL.toFixed(0)}°`, colorL);
  if (isFinite(angles.abdR)) labelAt(RH, `${angles.abdR.toFixed(0)}°`, colorR);
}

function updateDebug(L){
  if (!L){ dbg.textContent = 'Laukiu pozos…'; return; }
  const fmt = (p)=>`(${p.x.toFixed(3)}, ${p.y.toFixed(3)}, v=${(p.visibility??1).toFixed(2)})`;
  dbg.textContent =
    `LS ${fmt(L.leftShoulder)}  RS ${fmt(L.rightShoulder)}\n`+
    `LH ${fmt(L.leftHip)}  RH ${fmt(L.rightHip)}\n`+
    `LK ${fmt(L.leftKnee)} RK ${fmt(L.rightKnee)}`;
}

// ===== Pagrindinis ciklas =====
async function loop(){
  if (!pose || !video.videoWidth){ requestAnimationFrame(loop); return; }
  const out = await pose.detectForVideo(video, performance.now());

  if (out.landmarks && out.landmarks.length>0){
    const lm = out.landmarks[0];

    const L = {
      leftShoulder: lm[LM.LEFT_SHOULDER], rightShoulder: lm[LM.RIGHT_SHOULDER],
      leftHip: lm[LM.LEFT_HIP], rightHip: lm[LM.RIGHT_HIP],
      leftKnee: lm[LM.LEFT_KNEE], rightKnee: lm[LM.RIGHT_KNEE]
    };
    updateDebug(L);

    const ok = visOK(L.leftShoulder) && visOK(L.rightShoulder) &&
               visOK(L.leftHip) && visOK(L.rightHip) &&
               visOK(L.leftKnee) && visOK(L.rightKnee);

    if (ok){
      // BENDRA su manual app: S_mid → H_mid
      const midline = bodyMidlineFromLandmarks(L);
      const abdL = abductionPerHip2D(L.leftHip,  L.leftKnee,  midline.midDown);
      const abdR = abductionPerHip2D(L.rightHip, L.rightKnee, midline.midDown);

      drawOverlay(L, {abdL, abdR}, midline);

      const avg = (abdL + abdR)/2;
      avgVal.textContent = isFinite(avg)? `${avg.toFixed(1)}°` : '–';
      abdLVal.textContent = isFinite(abdL)? `${abdL.toFixed(0)}°` : '–';
      abdRVal.textContent = isFinite(abdR)? `${abdR.toFixed(0)}°` : '–';

      avgVal.style.color = colAbd(avg);
      abdLVal.style.color = colAbd(abdL);
      abdRVal.style.color = colAbd(abdR);
      dot.className = 'status-dot dot-active';

      if (avg > SAFE.yellowMax)       warn.textContent = '⚠️ Per didelė abdukcija (>60°).';
      else if (avg < SAFE.greenMin)   warn.textContent = '⚠️ Per maža abdukcija (<30°).';
      else if (avg <= SAFE.greenMax)  warn.textContent = 'Poza gera (30–45°).';
      else                             warn.textContent = 'Įspėjimas: 45–60° (geltona zona).';

      latest.ok = true;
      latest.angles = { abdL, abdR, avg };
      latest.midline = midline;
      latest.lms = L;

    } else {
      drawOverlay(null,{}, {S_mid:null,H_mid:null,midDown:null});
      avgVal.textContent='–'; abdLVal.textContent='–'; abdRVal.textContent='–';
      avgVal.style.color='#e5e7eb'; abdLVal.style.color='#e5e7eb'; abdRVal.style.color='#e5e7eb';
      dot.className = 'status-dot dot-idle';
      warn.textContent = 'Žemas matomumas – pataisyk kamerą/ar apšvietimą.';
      latest.ok = false;
    }
  } else {
    drawOverlay(null,{}, {S_mid:null,H_mid:null,midDown:null});
    label.textContent = 'Poza nerasta';
    dot.className = 'status-dot dot-idle';
    latest.ok = false;
  }
  requestAnimationFrame(loop);
}

// ===== Įrašymas kas 10 ms (tikslus laikas 0.01, 0.02, …) =====
function startCollect(){
  if (isCollecting) return;
  isCollecting = true;
  collectedData = [];
  sampleIdx = 0;
  bar.style.width = '0%';
  dot.className = 'status-dot dot-rec';
  label.textContent = 'Įrašau (2 s)…';
  dlBtn.disabled = true;

  // Įspėjimas jei telefonas pakrypęs >5°
  if (tiltDeg != null && Math.abs(tiltDeg) > 5){
    const proceed = confirm(`Telefonas pakreiptas ${tiltDeg.toFixed(1)}° (>5°).\nAr tikrai norite tęsti įrašą?`);
    if (!proceed){ stopCollect(); return; }
  }

  sampler = setInterval(()=>{
    const prog = ((sampleIdx+1)/RECORD_SAMPLES)*100;
    bar.style.width = `${Math.min(100,prog)}%`;

    if (sampleIdx >= RECORD_SAMPLES){
      stopCollect();
      return;
    }
    if (!latest.ok || !latest.angles || !latest.midline || !latest.lms){
      sampleIdx++; // laikas vis tiek „eina“, kad būtų tikslūs žingsniai
      return;
    }

    // tikslus „time“: 0.01, 0.02, ...
    const timeSec = +(((sampleIdx+1)*SAMPLE_MS)/1000).toFixed(2); // s

    const L = latest.lms;
    const md = latest.midline;
    const A = latest.angles;

    // JSON struktūra tokia pati kaip manual app (plius „time“)
    collectedData.push({
      timestamp: Date.now(),
      time: timeSec,
      patientCode: patientCode || null,
      angles: {
        abductionLeft:  +A.abdL.toFixed(2),
        abductionRight: +A.abdR.toFixed(2)
      },
      device: {
        tiltDeg: tiltDeg == null ? null : +tiltDeg.toFixed(2),
        tiltOK: tiltDeg == null ? null : (Math.abs(tiltDeg) <= 5)
      },
      midline: {
        from: { x:+md.S_mid.x.toFixed(4), y:+md.S_mid.y.toFixed(4) },
        to:   { x:+md.H_mid.x.toFixed(4), y:+md.H_mid.y.toFixed(4) }
      },
      midlineOffset: { dx: 0, dy: 0 }, // (šiame app’e rankinio poslinkio nėra)
      landmarks: {
        leftShoulder:  { x:L.leftShoulder.x,  y:L.leftShoulder.y,  v:L.leftShoulder.visibility??1 },
        rightShoulder: { x:L.rightShoulder.x, y:L.rightShoulder.y, v:L.rightShoulder.visibility??1 },
        leftHip:       { x:L.leftHip.x,       y:L.leftHip.y,       v:L.leftHip.visibility??1 },
        rightHip:      { x:L.rightHip.x,      y:L.rightHip.y,      v:L.rightHip.visibility??1 },
        leftKnee:      { x:L.leftKnee.x,      y:L.leftKnee.y,      v:L.leftKnee.visibility??1 },
        rightKnee:     { x:L.rightKnee.x,     y:L.rightKnee.y,     v:L.rightKnee.visibility??1 }
      }
    });

    sampleIdx++;
  }, SAMPLE_MS);
}

function stopCollect(){
  if (sampler){ clearInterval(sampler); sampler = null; }
  isCollecting = false;
  dot.className = 'status-dot dot-active';
  label.textContent = collectedData.length ? `Išsaugota ${collectedData.length} mėginių` : 'Įrašas be mėginių';
  dlBtn.disabled = collectedData.length===0;
}

function downloadJSON(){
  if (!collectedData.length) return;
  const blob = new Blob([JSON.stringify(collectedData,null,2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `pose_data_${new Date().toISOString().replace(/:/g,'-')}.json`;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ===== Events =====
startBtn.addEventListener('click', startCollect);
dlBtn.addEventListener('click', downloadJSON);
startCamBtn?.addEventListener('click', initCamera);

document.addEventListener('DOMContentLoaded', ()=>{
  perm.style.display = 'block'; // reikalingas vartotojo gestas kamerai paleisti
});
