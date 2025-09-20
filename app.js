import { PoseLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/+esm';

// UI
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

// --- įrašymo parametrai ---
const RECORD_MS = 2000;  // bendra trukmė: 2 s
const SAMPLE_MS = 10;    // mėginys kas 10 ms

let collectedData = [];
let isCollecting = false;
let t0 = 0;
let sampler = null; // setInterval rankenėlė

// --- MediaPipe / landmarks ---
let pose;
const LM = {
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26
};

// --- vektoriai / kampai ---
const unit = (v)=>{ const n=Math.hypot(v.x,v.y); return n>1e-6?{x:v.x/n,y:v.y/n}:{x:0,y:0}; };
const sub  = (a,b)=>({x:a.x-b.x, y:a.y-b.y});
const dotp = (a,b)=>a.x*b.x + a.y*b.y;
const ang  = (a,b)=>{ const c=Math.max(-1,Math.min(1, dotp(unit(a),unit(b)))); return Math.acos(c)*180/Math.PI; };

function pelvisBasis2D(LH, RH){
  const x = unit(sub(RH,LH));
  let midDown = unit({x:-x.y, y:x.x});      // statmena dubeniui
  if (midDown.y < 0) midDown = {x:-midDown.x, y:-midDown.y}; // nukreipiam žemyn
  return { x, midDown };
}
function abductionPerHip2D(HIP, KNEE, midDown){
  return ang(sub(KNEE, HIP), midDown); // kampas tarp šlaunies ir dubens „žemyn“ vektoriaus
}

// --- glodinimas: median(5) + EMA(α=0.12) + šuolio ribojimas ---
const SAFE = { abdMin:30, abdMax:45, abdWarnMax:55 };
const EMA_A = 0.12, JUMP = 10;
const qL=[], qR=[];
let emaL=null, emaR=null;

function pushQ(q, v){ q.push(v); if (q.length>5) q.shift(); }
function median(q){ const s=[...q].sort((a,b)=>a-b); return s[Math.floor(s.length/2)]; }
function smoothAngles(Lraw,Rraw){
  pushQ(qL, Lraw); pushQ(qR, Rraw);
  const Lm = (qL.length>=3)? median(qL) : Lraw;
  const Rm = (qR.length>=3)? median(qR) : Rraw;
  if (emaL!=null && Math.abs(Lm-emaL)>JUMP) return {L:emaL, R:emaR??Rm};
  if (emaR!=null && Math.abs(Rm-emaR)>JUMP) return {L:emaL??Lm, R:emaR};
  emaL = (emaL==null)? Lm : EMA_A*Lm + (1-EMA_A)*emaL;
  emaR = (emaR==null)? Rm : EMA_A*Rm + (1-EMA_A)*emaR;
  return { L:emaL, R:emaR };
}
const colAbd = (a)=> a>=SAFE.abdMin && a<=SAFE.abdMax ? '#34a853'
                    : a>SAFE.abdMax && a<=SAFE.abdWarnMax ? '#f9ab00'
                    : '#ea4335';

// --- kamera ---
async function initCamera(){
  try{
    let constraints = { video:{facingMode:{ideal:'environment'}, width:{ideal:1280}, height:{ideal:720}}, audio:false };
    let stream = await navigator.mediaDevices.getUserMedia(constraints).catch(()=>null);
    if (!stream){ stream = await navigator.mediaDevices.getUserMedia({video:true, audio:false}); }

    video.setAttribute('playsinline',''); video.setAttribute('muted',''); video.setAttribute('autoplay','');
    video.srcObject = stream;

    await new Promise(res => { if (video.readyState>=1) res(); else video.onloadedmetadata=res; });
    await video.play();

    perm.style.display = 'none';
    label.textContent = 'Kamera aktyvi';
    resizeCanvas();

    if (!pose) await initPose();
    startBtn.disabled = false;
    requestAnimationFrame(loop);
  } catch(err){
    console.error('Kameros klaida:', err);
    perm.style.display = 'block';
    label.textContent = 'Klaida: nepavyko pasiekti kameros';
    warn.textContent = 'Patikrink naršyklės leidimus (Settings > Browser > Camera: Allow).';
  }
}

function resizeCanvas(){
  // 1:1 su rodomu video dydžiu – kad taškai sėstų tiksliai ant vaizdo
  const w = video.clientWidth, h = video.clientHeight;
  if (!w || !h) return;
  canvas.width = w; canvas.height = h;
}
window.addEventListener('resize', resizeCanvas);

// --- MediaPipe init ---
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
const visOK = (lm)=> (lm.visibility ?? 1) >= 0.6;

// --- paskutinė būsena „sampleriui“ ---
let latest = {
  ok:false,
  angles:null,
  pelvisMidDown:null,
  lms:null
};

// --- minimalus overlay: pečiai/klubai/keliai taškai + HIP→KNEE linijos + vidurio linija ---
function drawOverlay(L, angles){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!L) return;

  const toPx = (p)=>({ x:p.x*canvas.width, y:p.y*canvas.height });

  // pixel koordinatės (tik reikalingi 6 taškai)
  const LS = toPx(L.leftShoulder),  RS = toPx(L.rightShoulder);
  const LH = toPx(L.leftHip),       RH = toPx(L.rightHip);
  const LK = toPx(L.leftKnee),      RK = toPx(L.rightKnee);

  // pelvis midline
  const basis = pelvisBasis2D(L.leftHip, L.rightHip);
  const mid = { x: ((L.leftHip.x+L.rightHip.x)/2)*canvas.width,
                y: ((L.leftHip.y+L.rightHip.y)/2)*canvas.height };
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(mid.x, mid.y);
  ctx.lineTo(mid.x + basis.midDown.x*120, mid.y + basis.midDown.y*120);
  ctx.stroke();

  // HIP→KNEE linijos (balta)
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 5; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(LH.x,LH.y); ctx.lineTo(LK.x,LK.y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(RH.x,RH.y); ctx.lineTo(RK.x,RK.y); ctx.stroke();

  // Taškai: pečiai, klubai, keliai (balti, maži)
  ctx.fillStyle = '#ffffff';
  for (const p of [LS, RS, LH, RH, LK, RK]){
    ctx.beginPath(); ctx.arc(p.x,p.y,4,0,Math.PI*2); ctx.fill();
  }

  // Kampo etiketės TIESIAI prie klubo taškų
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

// --- pagrindinis ciklas ---
async function loop(){
  if (!pose || !video.videoWidth){ requestAnimationFrame(loop); return; }
  const out = await pose.detectForVideo(video, performance.now());

  if (out.landmarks && out.landmarks.length>0){
    const lms = out.landmarks[0];

    const L = {
      leftShoulder: lms[LM.LEFT_SHOULDER], rightShoulder: lms[LM.RIGHT_SHOULDER],
      leftHip: lms[LM.LEFT_HIP], rightHip: lms[LM.RIGHT_HIP],
      leftKnee: lms[LM.LEFT_KNEE], rightKnee: lms[LM.RIGHT_KNEE]
    };
    updateDebug(L);

    const ok = visOK(L.leftShoulder) && visOK(L.rightShoulder) &&
               visOK(L.leftHip) && visOK(L.rightHip) &&
               visOK(L.leftKnee) && visOK(L.rightKnee);

    if (ok){
      const { midDown } = pelvisBasis2D(L.leftHip, L.rightHip);
      const rawL = abductionPerHip2D(L.leftHip, L.leftKnee, midDown);
      const rawR = abductionPerHip2D(L.rightHip, L.rightKnee, midDown);
      const { L:abdL, R:abdR } = smoothAngles(rawL, rawR);

      drawOverlay(L, {abdL, abdR});

      const avg = (abdL + abdR)/2;
      avgVal.textContent = isFinite(avg)? `${avg.toFixed(1)}°` : '–';
      abdLVal.textContent = isFinite(abdL)? `${abdL.toFixed(0)}°` : '–';
      abdRVal.textContent = isFinite(abdR)? `${abdR.toFixed(0)}°` : '–';

      const color = colAbd(avg);
      avgVal.style.color = color; abdLVal.style.color = color; abdRVal.style.color = color;
      dot.className = 'status-dot dot-active';

      if (avg > SAFE.abdWarnMax)      warn.textContent = '⚠️ Per didelė abdukcija (>55°).';
      else if (avg < SAFE.abdMin)     warn.textContent = '⚠️ Per maža abdukcija (<30°).';
      else                             warn.textContent = 'Viskas ribose.';

      // atnaujinam "latest" sampleriui
      latest.ok = true;
      latest.angles = { abdL, abdR, avg };
      latest.pelvisMidDown = midDown;
      latest.lms = {
        leftShoulder: L.leftShoulder, rightShoulder: L.rightShoulder,
        leftHip: L.leftHip, rightHip: L.rightHip,
        leftKnee: L.leftKnee, rightKnee: L.rightKnee
      };

    } else {
      drawOverlay(null,{});
      avgVal.textContent='–'; abdLVal.textContent='–'; abdRVal.textContent='–';
      avgVal.style.color='#e5e7eb'; abdLVal.style.color='#e5e7eb'; abdRVal.style.color='#e5e7eb';
      dot.className = 'status-dot dot-idle';
      warn.textContent = 'Žemas matomumas – pataisyk kamerą/ar apšvietimą.';
      latest.ok = false;
    }
  } else {
    drawOverlay(null,{});
    label.textContent = 'Poza nerasta';
    dot.className = 'status-dot dot-idle';
    latest.ok = false;
  }
  requestAnimationFrame(loop);
}

// --- įrašymas kas 10 ms, 2 sekundes ---
function startCollect(){
  if (isCollecting) return;
  isCollecting = true; t0 = Date.now();
  collectedData = [];
  bar.style.width = '0%';
  dot.className = 'status-dot dot-rec';
  label.textContent = 'Įrašau (2 s)…';
  dlBtn.disabled = true;

  // kas 10 ms – imame mėginį iš "latest"
  sampler = setInterval(()=>{
    const t = Date.now() - t0;
    const prog = (t/RECORD_MS)*100; bar.style.width = `${Math.min(100,prog)}%`;

    if (t > RECORD_MS){
      stopCollect();
      return;
    }
    if (!latest.ok || !latest.angles) return;

    const L = latest.lms;
    const md = latest.pelvisMidDown;
    const A = latest.angles;

    collectedData.push({
      timestamp: Date.now(),
      time: t,
      angles: {
        abductionAvg: +A.avg.toFixed(2),
        abductionLeft: +A.abdL.toFixed(2),
        abductionRight: +A.abdR.toFixed(2)
      },
      pelvis: { midDownX: md.x, midDownY: md.y },
      landmarks: {
        leftShoulder: { x:L.leftShoulder.x, y:L.leftShoulder.y, v:L.leftShoulder.visibility??1 },
        rightShoulder:{ x:L.rightShoulder.x,y:L.rightShoulder.y,v:L.rightShoulder.visibility??1 },
        leftHip:      { x:L.leftHip.x,      y:L.leftHip.y,      v:L.leftHip.visibility??1 },
        rightHip:     { x:L.rightHip.x,     y:L.rightHip.y,     v:L.rightHip.visibility??1 },
        leftKnee:     { x:L.leftKnee.x,     y:L.leftKnee.y,     v:L.leftKnee.visibility??1 },
        rightKnee:    { x:L.rightKnee.x,    y:L.rightKnee.y,    v:L.rightKnee.visibility??1 }
      }
    });
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

// events
startBtn.addEventListener('click', startCollect);
dlBtn.addEventListener('click', downloadJSON);
startCamBtn?.addEventListener('click', initCamera);

document.addEventListener('DOMContentLoaded', ()=>{
  perm.style.display = 'block'; // reikalingas vartotojo gestas kamerai paleisti
});
