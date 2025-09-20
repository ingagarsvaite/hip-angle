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

// time-series
let collectedData = [];
let isCollecting = false;
let t0 = 0;
const totalMs = 3000, saveStart = 1000, saveEnd = 2000;

// model / landmarks
let pose;
const LM = { LEFT_HIP:23, RIGHT_HIP:24, LEFT_KNEE:25, RIGHT_KNEE:26 };

// vec helpers
const unit = (v)=>{ const n=Math.hypot(v.x,v.y); return n>1e-6?{x:v.x/n,y:v.y/n}:{x:0,y:0}; };
const sub = (a,b)=>({x:a.x-b.x, y:a.y-b.y});
const dotp = (a,b)=>a.x*b.x + a.y*b.y;
const ang  = (a,b)=>{ const c=Math.max(-1,Math.min(1, dotp(unit(a),unit(b)))); return Math.acos(c)*180/Math.PI; };

function pelvisBasis2D(LH, RH){
  const x = unit(sub(RH,LH));
  let midDown = unit({x:-x.y, y:x.x});
  if (midDown.y < 0) midDown = {x:-midDown.x, y:-midDown.y};
  return { x, midDown };
}
function abductionPerHip2D(HIP, KNEE, midDown){
  return ang(sub(KNEE, HIP), midDown);
}

// smoothing (angles only): median(5) + EMA(α=0.12) + clamp
const SAFE = { abdMin:30, abdMax:45, abdWarnMax:55 };
const EMA_A = 0.12, JUMP = 10;
const qL=[], qR=[];
let emaL=null, emaR=null;

function pushQ(q, v){ q.push(v); if (q.length>5) q.shift(); }
function median(q){ const s=[...q].sort((a,b)=>a-b); const m=s[Math.floor(s.length/2)]; return m; }
function smoothAngles(Lraw,Rraw){
  pushQ(qL, Lraw); pushQ(qR, Rraw);
  const Lm = (qL.length>=3)? median(qL) : Lraw;
  const Rm = (qR.length>=3)? median(qR) : Rraw;
  // outlier clamp vs previous EMA
  if (emaL!=null && Math.abs(Lm-emaL)>JUMP) return {L:emaL, R:emaR??Rm};
  if (emaR!=null && Math.abs(Rm-emaR)>JUMP) return {L:emaL??Lm, R:emaR};
  emaL = (emaL==null)? Lm : EMA_A*Lm + (1-EMA_A)*emaL;
  emaR = (emaR==null)? Rm : EMA_A*Rm + (1-EMA_A)*emaR;
  return { L:emaL, R:emaR };
}

// colors
const colAbd = (a)=> a>=SAFE.abdMin && a<=SAFE.abdMax ? '#34a853'
                    : a>SAFE.abdMax && a<=SAFE.abdWarnMax ? '#f9ab00'
                    : '#ea4335';

// camera
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
  const wrap = document.getElementById('video-wrap');
  if (!video.videoWidth) return;
  canvas.width = wrap.clientWidth;
  canvas.height = wrap.clientWidth * (video.videoHeight / video.videoWidth);
}
window.addEventListener('resize', resizeCanvas);

// model
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
const toPx = (p)=>({ x:p.x*canvas.width, y:p.y*canvas.height });

// overlay (minimal)
function drawOverlay(L, angles){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!L) return;

  const basis = pelvisBasis2D(L.leftHip, L.rightHip);
  const mid = { x: ((L.leftHip.x+L.rightHip.x)/2)*canvas.width,
                y: ((L.leftHip.y+L.rightHip.y)/2)*canvas.height };

  // pelvis midline (white)
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(mid.x, mid.y);
  ctx.lineTo(mid.x + basis.midDown.x*120, mid.y + basis.midDown.y*120);
  ctx.stroke();

  // thighs (white)
  const LH=toPx(L.leftHip), RH=toPx(L.rightHip), LK=toPx(L.leftKnee), RK=toPx(L.rightKnee);
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 5; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(LH.x,LH.y); ctx.lineTo(LK.x,LK.y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(RH.x,RH.y); ctx.lineTo(RK.x,RK.y); ctx.stroke();

  // small dots exactly on HIP/KNEE (white)
  ctx.fillStyle = '#ffffff';
  for (const p of [LH,LK,RH,RK]){ ctx.beginPath(); ctx.arc(p.x,p.y,4,0,Math.PI*2); ctx.fill(); }

  // tiny arcs in color (per hip)
  function arcAt(HIPpx, deg, color){
    if (!isFinite(deg)) return;
    ctx.save();
    ctx.translate(HIPpx.x, HIPpx.y);
    ctx.strokeStyle = color; ctx.lineWidth = 3;
    const r = 32;
    ctx.beginPath();
    ctx.arc(0,0,r, Math.PI*0.5, Math.PI*0.5 + (deg*Math.PI/180), false);
    ctx.stroke();
    ctx.restore();
  }
  arcAt(LH, angles.abdL, colAbd(angles.abdL));
  arcAt(RH, angles.abdR, colAbd(angles.abdR));
}

function updateDebug(L){
  if (!L){ dbg.textContent = 'Laukiu pozos…'; return; }
  const fmt = (p)=>`(${p.x.toFixed(3)}, ${p.y.toFixed(3)}, v=${(p.visibility??1).toFixed(2)})`;
  dbg.textContent =
    `LH ${fmt(L.leftHip)}  RH ${fmt(L.rightHip)}\n`+
    `LK ${fmt(L.leftKnee)} RK ${fmt(L.rightKnee)}`;
}

// loop
async function loop(){
  if (!pose || !video.videoWidth){ requestAnimationFrame(loop); return; }
  const out = await pose.detectForVideo(video, performance.now());

  if (out.landmarks && out.landmarks.length>0){
    const lms = out.landmarks[0];
    const L = {
      leftHip: lms[LM.LEFT_HIP], rightHip: lms[LM.RIGHT_HIP],
      leftKnee: lms[LM.LEFT_KNEE], rightKnee: lms[LM.RIGHT_KNEE]
    };
    updateDebug(L);

    const ok = visOK(L.leftHip) && visOK(L.rightHip) && visOK(L.leftKnee) && visOK(L.rightKnee);

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

      if (isCollecting){
        const t = Date.now() - t0;
        const prog = (t/totalMs)*100; bar.style.width = `${Math.min(100,prog)}%`;
        if (t>=totalMs){ stopCollect(); }
        else if (t>=saveStart && t<=saveEnd){
          collectedData.push({
            timestamp: Date.now(),
            time: t,
            angles: { abductionAvg:+avg.toFixed(2), abductionLeft:+abdL.toFixed(2), abductionRight:+abdR.toFixed(2) },
            pelvis: { midDownX: midDown.x, midDownY: midDown.y },
            landmarks: {
              leftHip: {x:L.leftHip.x, y:L.leftHip.y, v:L.leftHip.visibility??1},
              rightHip:{x:L.rightHip.x,y:L.rightHip.y,v:L.rightHip.visibility??1},
              leftKnee:{x:L.leftKnee.x,y:L.leftKnee.y,v:L.leftKnee.visibility??1},
              rightKnee:{x:L.rightKnee.x,y:L.rightKnee.y,v:L.rightKnee.visibility??1}
            }
          });
        }
      } else {
        bar.style.width = '0%';
      }

      startBtn.disabled = false;

    } else {
      drawOverlay(null,{});
      avgVal.textContent='–'; abdLVal.textContent='–'; abdRVal.textContent='–';
      avgVal.style.color='#e5e7eb'; abdLVal.style.color='#e5e7eb'; abdRVal.style.color='#e5e7eb';
      dot.className = 'status-dot dot-idle';
      warn.textContent = 'Žemas matomumas – pataisyk kamerą/ar apšvietimą.';
    }
  } else {
    drawOverlay(null,{});
    label.textContent = 'Poza nerasta';
    dot.className = 'status-dot dot-idle';
  }
  requestAnimationFrame(loop);
}

// record/export
function startCollect(){
  if (isCollecting) return;
  isCollecting = true; t0 = Date.now();
  collectedData = [];
  dot.className = 'status-dot dot-rec';
  label.textContent = 'Įrašau (3 s)…';
  bar.style.width = '0%';
  dlBtn.disabled = true;
}
function stopCollect(){
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
  perm.style.display = 'block'; // require explicit gesture on all devices
});
