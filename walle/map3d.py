"""Interactive 3D map viewer: writes ONE self-contained HTML file per mission (no libraries, works offline).

Shows dark tiles, glowing obstacle blocks, the belief as rising 3D columns (heatmap), the rover trail,
revealed clues, optional ground truth, heartbeat ping rings on every scan, and play/pause/step/speed controls.
Drag to rotate, wheel to zoom.
"""
import json
import numpy as np


def _frames_and_beliefs(env, trace, snaps, report):
    beliefs, bsteps = [], []
    for st, p in snaps:
        t = np.round(p / max(float(p.max()), 1e-12), 3)
        if beliefs and np.array_equal(beliefs[-1], t):
            continue
        beliefs.append(t)
        bsteps.append(st)

    def bidx(step):
        j = 0
        for i, s in enumerate(bsteps):
            if s <= step:
                j = i
        return j

    frames = [{"step": 0, "pos": list(env.start), "type": "start", "decision": "start",
               "battery": env.cfg["budget"]["battery"], "time": env.cfg["budget"]["time"], "conf": 0.0, "entropy": 0.0,
               "reasoning": "Mission start. Only the weak initial signal is known; the belief is a broad cloud around it.",
               "b": 0, "reveal": [], "reading": None}]
    for r in trace:
        sc = r["observation"].get("scan") if r["observation"] else None
        frames.append({"step": r["step"], "pos": r["pos_after"], "type": r["action"]["type"], "decision": r["decision"],
                       "battery": r["battery"], "time": r["time_left"], "conf": r["belief"]["confidence"],
                       "entropy": r["belief"]["entropy"], "reasoning": r["reasoning"], "b": bidx(r["step"]),
                       "reveal": [c["id"] for c in sc["clues"]] if sc else [],
                       "reading": sc["reading"] if sc else None})
    return frames, [b.tolist() for b in beliefs]


def export_map3d(path, env, trace, snaps, report, cfg):
    frames, beliefs = _frames_and_beliefs(env, trace, snaps, report)
    tr = env.truth()
    data = {
        "name": cfg["name"], "W": env.W, "H": env.H,
        "walls": [[int(x), int(y)] for y, x in np.argwhere(~env.free)],
        "start": list(env.start), "eve": list(tr["eve_pos"]), "eveStatus": tr["eve_status"],
        "clues": [{"id": c["id"], "pos": list(c["pos"]), "kind": c["kind"], "reliable": c["reliable"],
                   "estimate": c["estimate"]} for c in tr["clues"]],
        "frames": frames, "beliefs": beliefs, "battFull": cfg["budget"]["battery"],
        "status": report["status"], "inferred": report["inferred_location"], "confidence": report["confidence"],
        "scanRange": 2.2 * cfg["sensor"]["length_scale"],
    }
    js = json.dumps(data).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(HTML.replace("__DATA__", js).replace("__TITLE__", cfg["name"]))


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>3D Map - __TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;height:100%;background:#050b1e;color:#cfe3ff;font:13px/1.4 "Segoe UI",system-ui,sans-serif;overflow:hidden}
#app{display:flex;flex-direction:column;height:100%}
#bar{display:flex;gap:8px;align-items:center;padding:8px 12px;background:#0a1330;border-bottom:1px solid #1c2c63;flex-wrap:wrap}
.title{font-weight:800;letter-spacing:1px;color:#ffc400;margin-right:6px}.sep{width:1px;height:22px;background:#1c2c63;margin:0 4px}
.chip{border:1px solid #2a3f8a;padding:4px 11px;border-radius:14px;background:#0f1d4a;color:#9fc0ff;user-select:none;cursor:pointer}
.chip:hover{border-color:#22d3ee}.chip.on{background:#0e7490;border-color:#22d3ee;color:#fff}
.chip.gt.on{background:#6d28d9;border-color:#a78bfa}
#stage{position:relative;flex:1;min-height:260px;background:radial-gradient(ellipse at 50% 30%,#16286a 0%,#070d26 72%)}
canvas{position:absolute;inset:0;width:100%;height:100%;cursor:grab;touch-action:none}canvas:active{cursor:grabbing}
#legend{position:absolute;left:10px;bottom:8px;font-size:11px;color:#8fb0ee;background:#0a133099;padding:6px 9px;border-radius:6px;line-height:1.6;pointer-events:none}
#legend i{display:inline-block;width:10px;height:10px;margin-right:6px;vertical-align:-1px;border-radius:2px}
#badge{position:absolute;right:10px;top:8px;font-weight:800;letter-spacing:2px;padding:4px 12px;border:2px solid #22d3ee;color:#22d3ee;border-radius:4px;display:none;background:#0a1330cc}
#ctl{display:flex;gap:8px;align-items:center;padding:8px 12px;background:#0a1330;border-top:1px solid #1c2c63;flex-wrap:wrap}
button{background:#12306b;color:#dbe8ff;border:1px solid #2d55b3;padding:6px 12px;border-radius:6px;cursor:pointer;font:inherit}
button:hover{background:#1d4ed8}button.go{background:#059669;border-color:#10b981}
input[type=range]{flex:1;min-width:140px;accent-color:#22d3ee}
#info{padding:7px 14px 9px;background:#070e24;border-top:1px solid #1c2c63;display:flex;gap:18px;align-items:flex-start}
#info .stats{min-width:210px;font-variant-numeric:tabular-nums;color:#9fc0ff}#info .stats b{color:#fff}
#why{flex:1;color:#cfe3ff;max-height:64px;overflow:auto}.tag{display:inline-block;background:#0e7490;color:#fff;padding:0 7px;border-radius:3px;margin-right:6px;font-size:11px;font-weight:700}
#bb{display:inline-block;width:90px;height:8px;background:#13204f;border-radius:4px;vertical-align:middle;margin-left:6px;overflow:hidden}#bb i{display:block;height:100%;background:#22c55e}
</style></head><body><div id="app">
<div id="bar"><span class="title">WALL-E &middot; EVE SIGNAL AGENT &middot; 3D MAP</span>
  <span class="chip on" data-l="belief">Belief Heatmap</span><span class="chip on" data-l="trail">Rover Trail</span>
  <span class="chip on" data-l="clues">Clues</span><span class="chip gt" data-l="truth">Ground Truth</span><span class="sep"></span>
  <span class="chip" data-v="iso">Iso</span><span class="chip" data-v="top">Top</span><span class="chip" data-v="side">Low angle</span>
  <span class="chip" id="auto">Auto-rotate</span></div>
<div id="stage"><canvas id="c"></canvas><div id="badge"></div>
  <div id="legend"><i style="background:#ff7a1a"></i>obstacle &nbsp;<i style="background:#1d4ed8"></i>base/charger &nbsp;<i style="background:#22d3ee"></i>WALL-E &nbsp;<i style="background:#ffc400"></i>clue (found)<br>
  <i style="background:linear-gradient(90deg,#13307a,#18a0e6,#ffdc3c,#fff)"></i>belief: where EVE probably is (taller = likelier) &nbsp; drag = rotate, wheel = zoom</div></div>
<div id="ctl"><button id="rs" title="back to start">&#9198;</button><button id="bk">&#9664; Step</button><button id="pl" class="go">Pause</button><button id="fw">Step &#9654;</button>
  <span class="chip on" data-s="1">1x</span><span class="chip" data-s="2">2x</span><span class="chip" data-s="4">4x</span><span class="chip" data-s="0.5">0.5x</span>
  <input type="range" id="scrub" min="0" value="0"><span id="cnt" style="min-width:90px;text-align:right"></span></div>
<div id="info"><div class="stats" id="st"></div><div id="why"></div></div></div>
<script>
const D=__DATA__, W=D.W, H=D.H, F=D.frames;
const cv=document.getElementById('c'), ctx=cv.getContext('2d');
let yaw=0.75, pitch=0.95, zoom=1, auto=false, fi=0, playing=true, speed=1;
let lastAdv=performance.now(), frameStart=performance.now(), pw=1, ph=1, dpr=1, C;
const layers={belief:true,trail:true,clues:true,truth:false};
const wall=new Set(D.walls.map(w=>w[0]+','+w[1]));
const revealed=[]; {let s=new Set(); for(const f of F){f.reveal.forEach(i=>s.add(i)); revealed.push(new Set(s));}}
const clueById={}; D.clues.forEach(c=>clueById[c.id]=c);
const last=F.length-1, found=D.status==='CONFIRMED';
const scrub=document.getElementById('scrub'); scrub.max=last;

function resize(){const r=cv.getBoundingClientRect();dpr=window.devicePixelRatio||1;cv.width=Math.max(1,r.width*dpr);cv.height=Math.max(1,r.height*dpr);pw=r.width;ph=r.height}
addEventListener('resize',resize);
function mk(){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const ppu=0.58*Math.min(pw,ph*1.1)/Math.max(W,H)*zoom, d0=Math.max(W,H)*1.8; return {cy,sy,cp,sp,ppu,d0,f:d0*ppu}}
function P(gx,h,gy){const X=gx-(W-1)/2,Z=gy-(H-1)/2,x1=X*C.cy-Z*C.sy,z1=X*C.sy+Z*C.cy,y2=h*C.cp+z1*C.sp,z2=z1*C.cp-h*C.sp,s=C.f/(C.d0+z2);
  return [pw/2+x1*s, ph*0.56-y2*s, z2]}
function NZ(nx,ny,nz){const x1=nx*C.cy-nz*C.sy,z1=nx*C.sy+nz*C.cy;return [x1,z1*C.cp-ny*C.sp]}
const rgb=(c,k=1,a=1)=>`rgba(${Math.min(255,c[0]*k)|0},${Math.min(255,c[1]*k)|0},${Math.min(255,c[2]*k)|0},${a})`;
function ramp(t){const S=[[0,[19,48,122]],[.35,[24,160,230]],[.65,[255,220,60]],[1,[255,255,255]]];
  for(let i=1;i<S.length;i++)if(t<=S[i][0]){const u=(t-S[i-1][0])/(S[i][0]-S[i-1][0]),a=S[i-1][1],b=S[i][1];return a.map((v,j)=>v+(b[j]-v)*u)}return S[3][1]}
function poly(pts,fill,stroke,lw){ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();
  if(fill){ctx.fillStyle=fill;ctx.fill()} if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lw||1;ctx.stroke()}}
function box(gx,gy,y0,y1,hs,top,side,edge){
  const a=gx-hs,b=gx+hs,c=gy-hs,d=gy+hs;
  const faces=[[[1,0,0],[[b,y0,c],[b,y0,d],[b,y1,d],[b,y1,c]]],[[-1,0,0],[[a,y0,d],[a,y0,c],[a,y1,c],[a,y1,d]]],
    [[0,0,1],[[b,y0,d],[a,y0,d],[a,y1,d],[b,y1,d]]],[[0,0,-1],[[a,y0,c],[b,y0,c],[b,y1,c],[a,y1,c]]],[[0,1,0],[[a,y1,c],[b,y1,c],[b,y1,d],[a,y1,d]]]];
  for(const [n,q] of faces){const [x1,zn]=NZ(n[0],n[1],n[2]); if(zn>=0)continue;
    const pts=q.map(v=>P(v[0],v[1],v[2])); poly(pts, n[1]===1?rgb(top):rgb(side,0.62+0.30*x1), edge||'rgba(0,0,0,.35)',1)}}
function label(t,gx,h,gy,col,sz){const p=P(gx,h,gy);ctx.font=`bold ${sz||11}px Segoe UI,sans-serif`;ctx.textAlign='center';ctx.fillStyle=col||'#fff';ctx.fillText(t,p[0],p[1])}
function ring(gx,gy,r,col,lw,alpha,h){ctx.beginPath();for(let i=0;i<=48;i++){const a=i/48*6.2832,p=P(gx+Math.cos(a)*r,h||0.06,gy+Math.sin(a)*r);i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])}
  ctx.strokeStyle=col;ctx.globalAlpha=Math.max(0,alpha);ctx.lineWidth=lw;ctx.stroke();ctx.globalAlpha=1}
function diamond(gx,h,gy,sz,fill,stroke,txt){const p=P(gx,h,gy);ctx.beginPath();ctx.moveTo(p[0],p[1]-sz);ctx.lineTo(p[0]+sz*.75,p[1]);ctx.lineTo(p[0],p[1]+sz);ctx.lineTo(p[0]-sz*.75,p[1]);ctx.closePath();
  ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke;ctx.lineWidth=1.5;ctx.stroke();if(txt){ctx.font='bold 11px sans-serif';ctx.textAlign='center';ctx.fillStyle='#1a1200';ctx.fillText(txt,p[0],p[1]+4)}}

function draw(now){
  if(auto)yaw+=0.004;
  C=mk(); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,pw,ph);
  const f=F[fi], bel=D.beliefs[f.b], cur=f.pos;
  // ground tiles
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){ if(wall.has(x+','+y))continue;
    const q=[[x-.5,0,y-.5],[x+.5,0,y-.5],[x+.5,0,y+.5],[x-.5,0,y+.5]].map(v=>P(v[0],v[1],v[2]));
    const isB=(x===D.start[0]&&y===D.start[1]); poly(q,isB?'#1d4ed8':'#0d1a40','#1c2d66',1)}
  // sorted 3D objects
  const obj=[];
  for(const w of D.walls)obj.push({z:P(w[0],0,w[1])[2],fn:()=>{box(w[0],w[1],0,1.0,.46,[255,122,26],[190,80,10],'rgba(255,190,90,.55)');
    const q=[[w[0]-.2,1.0,w[1]-.2],[w[0]+.2,1.0,w[1]-.2],[w[0]+.2,1.0,w[1]+.2],[w[0]-.2,1.0,w[1]+.2]].map(v=>P(v[0],v[1],v[2]));poly(q,'rgba(255,225,140,.95)')}});
  if(layers.belief)for(let y=0;y<H;y++)for(let x=0;x<W;x++){const t=bel[y][x]; if(t<0.04||wall.has(x+','+y))continue;
    const h=0.08+3.4*Math.pow(t,0.9); obj.push({z:P(x,0,y)[2]+0.001,fn:()=>{const col=ramp(t);ctx.globalAlpha=.9;box(x,y,0,h,.42,col,col,'rgba(255,255,255,.18)');ctx.globalAlpha=1}})}
  obj.push({z:P(D.start[0],0,D.start[1])[2]-0.01,fn:()=>{box(D.start[0],D.start[1],0,.18,.44,[59,130,246],[30,64,175]);label('BASE',D.start[0],.5,D.start[1],'#dbeafe',10)}});
  const eveShown=layers.truth||(found&&fi===last);
  if(eveShown)obj.push({z:P(D.eve[0],0,D.eve[1])[2]+0.002,fn:()=>{box(D.eve[0],D.eve[1],0,3.0,.16,[255,60,60],[200,20,20],'rgba(255,255,255,.5)');label('EVE',D.eve[0],3.5,D.eve[1],'#ff9a9a',12)}});
  obj.push({z:P(cur[0],0,cur[1])[2]+0.003,fn:()=>{box(cur[0],cur[1],0,.9,.28,[34,211,238],[8,145,178],'#fff');label('W',cur[0],1.25,cur[1],'#fff',13)}});
  if(layers.clues){const rv=revealed[fi];for(const c of D.clues){const seen=rv.has(c.id);if(!seen&&!layers.truth)continue;
    obj.push({z:P(c.pos[0],0,c.pos[1])[2]+0.002,fn:()=>{
      if(seen)diamond(c.pos[0],1.15,c.pos[1],9,'#ffc400','#fff','?'); else diamond(c.pos[0],1.15,c.pos[1],8,'rgba(0,0,0,0)',c.reliable?'#4ade80':'#f87171');
      const g=P(c.pos[0],0.1,c.pos[1]),t=P(c.pos[0],1.0,c.pos[1]);ctx.beginPath();ctx.moveTo(g[0],g[1]);ctx.lineTo(t[0],t[1]);ctx.strokeStyle=seen?'#ffc400':(c.reliable?'#4ade80':'#f87171');ctx.lineWidth=1;ctx.stroke()}})}}
  obj.sort((a,b)=>b.z-a.z); obj.forEach(o=>o.fn());
  // overlays: clue pointers, trail, ping rings
  if(layers.clues){const rv=revealed[fi];ctx.setLineDash([5,4]);for(const id of rv){const c=clueById[id];const a=P(c.pos[0],.25,c.pos[1]),b=P(c.estimate[0],.25,c.estimate[1]);
    ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.strokeStyle=layers.truth?(c.reliable?'#4ade80':'#f87171'):'#ffc400';ctx.lineWidth=1.3;ctx.stroke();
    ctx.beginPath();ctx.arc(b[0],b[1],4,0,6.3);ctx.stroke()}ctx.setLineDash([])}
  if(layers.trail){ctx.beginPath();let first=true;for(let i=0;i<=fi;i++){const p=P(F[i].pos[0],.2,F[i].pos[1]);first?ctx.moveTo(p[0],p[1]):ctx.lineTo(p[0],p[1]);first=false}
    ctx.strokeStyle='rgba(34,211,238,.95)';ctx.lineWidth=3;ctx.shadowColor='#22d3ee';ctx.shadowBlur=8;ctx.stroke();ctx.shadowBlur=0;
    for(let i=0;i<=fi;i++)if(F[i].type==='scan'){const p=P(F[i].pos[0],.2,F[i].pos[1]);ctx.beginPath();ctx.arc(p[0],p[1],5,0,6.3);ctx.fillStyle='#ffa500';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke()}}
  if(f.type==='scan'){const t=(now-frameStart)/1100;if(t<1.3){const rd=Math.min(1,Math.max(0,f.reading||0)),base=.35+.65*rd;
    for(const lag of [0,.14,.28]){const u=t-lag;if(u>0&&u<1)ring(cur[0],cur[1],u*D.scanRange,'#39ff14',3.2-2*u,(1-u)*base)}}}
  if(eveShown&&found&&fi===last){const th=Math.abs(Math.sin(now/260));[.8,1.4,2.1].forEach((r,i)=>ring(D.eve[0],D.eve[1],r+th*(.4+.25*i),'#ff3030',2.4,.9-.22*i,.08))}
  { const g=P(cur[0],0.1,cur[1]), t=P(cur[0],2.6,cur[1]);
    ctx.beginPath();ctx.moveTo(g[0],g[1]);ctx.lineTo(t[0],t[1]);ctx.strokeStyle='rgba(34,211,238,.9)';ctx.lineWidth=2;ctx.shadowColor='#22d3ee';ctx.shadowBlur=10;ctx.stroke();
    ctx.beginPath();ctx.arc(t[0],t[1],10,0,6.3);ctx.fillStyle='#22d3ee';ctx.fill();ctx.shadowBlur=0;ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
    ctx.font='bold 12px Segoe UI,sans-serif';ctx.textAlign='center';ctx.fillStyle='#012';ctx.fillText('W',t[0],t[1]+4);}
  // HUD
  const bb=document.getElementById('badge');if(fi===last){bb.style.display='block';bb.textContent=D.status+' at ('+D.inferred.join(', ')+')';bb.style.color=bb.style.borderColor=found?'#ff6b6b':'#fbbf24'}else bb.style.display='none';
}
function hud(){const f=F[fi];document.getElementById('cnt').textContent='frame '+fi+' / '+last;scrub.value=fi;
  document.getElementById('st').innerHTML='step <b>'+f.step+'</b> &nbsp; battery <b>'+f.battery+'</b>/'+D.battFull+'<span id="bb"><i style="width:'+Math.round(100*f.battery/D.battFull)+'%"></i></span><br>confidence <b>'+Math.round(f.conf*100)+'%</b> &nbsp; time left <b>'+f.time+'</b>'+(f.reading!==null?'<br>signal return <b>'+f.reading.toFixed(2)+'</b>':'');
  document.getElementById('why').innerHTML='<span class="tag">'+f.decision.toUpperCase()+'</span>'+f.reasoning.replace(/&/g,'&amp;').replace(/</g,'&lt;');
  document.getElementById('pl').textContent=playing?'Pause':(fi===last?'Replay':'Play')}
function setFrame(n){fi=Math.max(0,Math.min(last,n));frameStart=lastAdv=performance.now();hud()}
function loop(now){ if(playing){const f=F[fi],d=(f.type==='scan'?1300:f.decision==='replan'?520:f.type==='declare'?800:240)/speed;
    if(now-lastAdv>d){if(fi<last)setFrame(fi+1);else{playing=false;hud()}}}
  draw(now);requestAnimationFrame(loop)}
// controls
document.querySelectorAll('[data-l]').forEach(e=>e.onclick=()=>{layers[e.dataset.l]=!layers[e.dataset.l];e.classList.toggle('on',layers[e.dataset.l])});
document.querySelectorAll('[data-v]').forEach(e=>e.onclick=()=>{const v=e.dataset.v;auto=false;document.getElementById('auto').classList.remove('on');
  if(v==='iso'){yaw=.75;pitch=.95}if(v==='top'){yaw=0;pitch=1.5}if(v==='side'){yaw=.35;pitch=.38}zoom=1});
document.getElementById('auto').onclick=e=>{auto=!auto;e.target.classList.toggle('on',auto)};
document.querySelectorAll('[data-s]').forEach(e=>e.onclick=()=>{speed=+e.dataset.s;document.querySelectorAll('[data-s]').forEach(x=>x.classList.toggle('on',x===e))});
document.getElementById('pl').onclick=()=>{if(!playing&&fi===last)setFrame(0);playing=!playing;lastAdv=performance.now();hud()};
document.getElementById('fw').onclick=()=>{playing=false;setFrame(fi+1)};document.getElementById('bk').onclick=()=>{playing=false;setFrame(fi-1)};
document.getElementById('rs').onclick=()=>{playing=false;setFrame(0)};scrub.oninput=()=>{playing=false;setFrame(+scrub.value)};
let drag=null;cv.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId)});
cv.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*0.008;pitch=Math.max(.2,Math.min(1.5,pitch+(e.clientY-drag[1])*0.006));drag=[e.clientX,e.clientY]});
cv.addEventListener('pointerup',()=>drag=null);cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.5,Math.min(3,zoom*(e.deltaY<0?1.1:.9)))},{passive:false});
window.setFrame=setFrame;window.setPlaying=v=>{playing=v;hud()};
resize();hud();requestAnimationFrame(loop);
</script></body></html>"""
