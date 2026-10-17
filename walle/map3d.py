"""Interactive 3D map viewer (blueprint / hologram ROOM style): writes ONE self-contained HTML file per mission (no libraries, works offline).

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


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>3D Room - __TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;height:100%;background:#06236f;color:#e8f4ff;font:13px/1.4 "Segoe UI",system-ui,sans-serif;overflow:hidden}
#app{display:flex;flex-direction:column;height:100%}
#bar{display:flex;gap:8px;align-items:center;padding:8px 12px;background:#07226b;border-bottom:1px solid #2f63d6;flex-wrap:wrap}
.title{font-weight:800;letter-spacing:1px;color:#ffd24a;margin-right:6px}.sep{width:1px;height:22px;background:#2f63d6;margin:0 4px}
.chip{border:1px solid #3d72e6;padding:4px 11px;border-radius:14px;background:#0d3190;color:#bcd7ff;user-select:none;cursor:pointer}
.chip:hover{border-color:#bff0ff}.chip.on{background:#1fb6e8;border-color:#bff0ff;color:#04204f;font-weight:600}
.chip.gt.on{background:#8b5cf6;border-color:#d4c4ff;color:#fff}
#stage{position:relative;flex:1;min-height:260px;background:radial-gradient(ellipse at 50% 38%,#2a7bf0 0%,#1253c9 38%,#08297a 100%)}
canvas{position:absolute;inset:0;width:100%;height:100%;cursor:grab;touch-action:none}canvas:active{cursor:grabbing}
#legend{position:absolute;left:10px;bottom:8px;font-size:11px;color:#d6e9ff;background:#06236fcc;padding:6px 9px;border-radius:6px;line-height:1.6;pointer-events:none}
#legend i{display:inline-block;width:10px;height:10px;margin-right:6px;vertical-align:-1px;border-radius:2px}
#badge{position:absolute;right:10px;top:8px;font-weight:800;letter-spacing:2px;padding:4px 12px;border:2px solid #ffd24a;color:#ffd24a;border-radius:4px;display:none;background:#06236fdd}
#ctl{display:flex;gap:8px;align-items:center;padding:8px 12px;background:#07226b;border-top:1px solid #2f63d6;flex-wrap:wrap}
button{background:#1650c8;color:#eaf4ff;border:1px solid #5b90ff;padding:6px 12px;border-radius:6px;cursor:pointer;font:inherit}
button:hover{background:#2f7bff}button.go{background:#0f9d6c;border-color:#34e0a1}
input[type=range]{flex:1;min-width:140px;accent-color:#7fe3ff}
#info{padding:7px 14px 9px;background:#051a54;border-top:1px solid #2f63d6;display:flex;gap:18px;align-items:flex-start}
#info .stats{min-width:210px;font-variant-numeric:tabular-nums;color:#bcd7ff}#info .stats b{color:#fff}
#why{flex:1;color:#e8f4ff;max-height:64px;overflow:auto}.tag{display:inline-block;background:#1fb6e8;color:#04204f;padding:0 7px;border-radius:3px;margin-right:6px;font-size:11px;font-weight:700}
#bb{display:inline-block;width:90px;height:8px;background:#0a2a7a;border-radius:4px;vertical-align:middle;margin-left:6px;overflow:hidden}#bb i{display:block;height:100%;background:#4ade80}
</style></head><body><div id="app">
<div id="bar"><span class="title">WALL-E &middot; EVE SIGNAL &middot; 3D ROOM</span>
  <span class="chip on" data-l="belief">Belief Heatmap</span><span class="chip on" data-l="trail">Rover Trail</span>
  <span class="chip on" data-l="clues">Clues</span><span class="chip gt" data-l="truth">Ground Truth</span><span class="sep"></span>
  <span class="chip" data-v="iso">Room view</span><span class="chip" data-v="top">Top</span><span class="chip" data-v="side">Low angle</span>
  <span class="chip" id="auto">Auto-rotate</span></div>
<div id="stage"><canvas id="c"></canvas><div id="badge"></div>
  <div id="legend"><i style="background:#ffb347"></i>crates (obstacles) &nbsp;<i style="background:#5be7ff"></i>base pad &nbsp;<i style="background:#ffc400"></i>WALL-E &nbsp;<i style="background:#ffd24a"></i>clue found<br>
  <i style="background:linear-gradient(90deg,#1b47b8,#18c0f0,#ffe35a,#fff)"></i>floor glow = where EVE probably is (brighter = likelier) &nbsp; drag = rotate, wheel = zoom</div></div>
<div id="ctl"><button id="rs" title="back to start">&#9198;</button><button id="bk">&#9664; Step</button><button id="pl" class="go">Pause</button><button id="fw">Step &#9654;</button>
  <span class="chip on" data-s="1">1x</span><span class="chip" data-s="2">2x</span><span class="chip" data-s="4">4x</span><span class="chip" data-s="0.5">0.5x</span>
  <input type="range" id="scrub" min="0" value="0"><span id="cnt" style="min-width:90px;text-align:right"></span></div>
<div id="info"><div class="stats" id="st"></div><div id="why"></div></div></div>
<script>
const D=__DATA__, W=D.W, H=D.H, F=D.frames, WH=2.8;
const cv=document.getElementById('c'), ctx=cv.getContext('2d');
let yaw=0.8, pitch=0.82, zoom=1, auto=false, fi=0, playing=true, speed=1;
let lastAdv=performance.now(), frameStart=performance.now(), pw=1, ph=1, dpr=1, C;
const layers={belief:true,trail:true,clues:true,truth:false};
const wall=new Set(D.walls.map(w=>w[0]+','+w[1]));
const revealed=[]; {let s=new Set(); for(const f of F){f.reveal.forEach(i=>s.add(i)); revealed.push(new Set(s));}}
const clueById={}; D.clues.forEach(c=>clueById[c.id]=c);
const last=F.length-1, found=D.status==='CONFIRMED';
const scrub=document.getElementById('scrub'); scrub.max=last;
const INK='rgba(232,246,255,', AMB='rgba(255,186,80,';

function resize(){const r=cv.getBoundingClientRect();dpr=window.devicePixelRatio||1;cv.width=Math.max(1,r.width*dpr);cv.height=Math.max(1,r.height*dpr);pw=r.width;ph=r.height}
addEventListener('resize',resize);
function mk(){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const ppu=0.56*Math.min(pw*1.0,ph*1.15)/Math.max(W,H)*zoom, d0=Math.max(W,H)*1.9; return {cy,sy,cp,sp,ppu,d0,f:d0*ppu}}
function P(gx,h,gy){const X=gx-(W-1)/2,Z=gy-(H-1)/2,x1=X*C.cy-Z*C.sy,z1=X*C.sy+Z*C.cy,y2=(h-WH*0.35)*C.cp+z1*C.sp,z2=z1*C.cp-(h-WH*0.35)*C.sp,s=C.f/(C.d0+z2);
  return [pw/2+x1*s, ph*0.49-y2*s, z2]}
function NZ(nx,ny,nz){const x1=nx*C.cy-nz*C.sy,z1=nx*C.sy+nz*C.cy;return [x1,z1*C.cp-ny*C.sp]}
const rgba=(c,a)=>`rgba(${c[0]|0},${c[1]|0},${c[2]|0},${a})`;
function ramp(t){const S=[[0,[27,71,184]],[.3,[24,192,240]],[.65,[255,227,90]],[1,[255,255,255]]];
  for(let i=1;i<S.length;i++)if(t<=S[i][0]){const u=(t-S[i-1][0])/(S[i][0]-S[i-1][0]),a=S[i-1][1],b=S[i][1];return a.map((v,j)=>v+(b[j]-v)*u)}return S[3][1]}
function poly(pts,fill,stroke,lw){ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();
  if(fill){ctx.fillStyle=fill;ctx.fill()} if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=lw||1;ctx.stroke()}}
function line3(a,b,stroke,lw){const p=P(a[0],a[1],a[2]),q=P(b[0],b[1],b[2]);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.strokeStyle=stroke;ctx.lineWidth=lw||1;ctx.stroke()}
function wbox(cx,cz,hx,hz,y0,y1,stroke,fill,lw){
  const x0=cx-hx,x1=cx+hx,z0=cz-hz,z1=cz+hz;
  const faces=[[[1,0,0],[[x1,y0,z0],[x1,y0,z1],[x1,y1,z1],[x1,y1,z0]]],[[-1,0,0],[[x0,y0,z1],[x0,y0,z0],[x0,y1,z0],[x0,y1,z1]]],
    [[0,0,1],[[x1,y0,z1],[x0,y0,z1],[x0,y1,z1],[x1,y1,z1]]],[[0,0,-1],[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]]],[[0,1,0],[[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]]]];
  for(const [n,q] of faces){const zn=NZ(n[0],n[1],n[2])[1]; if(zn>=0)continue; poly(q.map(v=>P(v[0],v[1],v[2])),fill,null)}
  const c=[[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1],[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]].map(v=>P(v[0],v[1],v[2]));
  const E=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  ctx.strokeStyle=stroke;ctx.lineWidth=lw||1.3;ctx.beginPath();for(const [i,j] of E){ctx.moveTo(c[i][0],c[i][1]);ctx.lineTo(c[j][0],c[j][1])}ctx.stroke()}
function label(t,gx,h,gy,col,sz){const p=P(gx,h,gy);ctx.font=`bold ${sz||11}px Segoe UI,sans-serif`;ctx.textAlign='center';ctx.fillStyle=col||'#fff';ctx.fillText(t,p[0],p[1])}
function ring(gx,gy,r,col,lw,alpha,h){ctx.beginPath();for(let i=0;i<=48;i++){const a=i/48*6.2832,p=P(gx+Math.cos(a)*r,h||0.04,gy+Math.sin(a)*r);i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])}
  ctx.strokeStyle=col;ctx.globalAlpha=Math.max(0,alpha);ctx.lineWidth=lw;ctx.stroke();ctx.globalAlpha=1}
function diamond(gx,h,gy,sz,fill,stroke,txt){const p=P(gx,h,gy);ctx.beginPath();ctx.moveTo(p[0],p[1]-sz);ctx.lineTo(p[0]+sz*.75,p[1]);ctx.lineTo(p[0],p[1]+sz);ctx.lineTo(p[0]-sz*.75,p[1]);ctx.closePath();
  ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke;ctx.lineWidth=1.5;ctx.stroke();if(txt){ctx.font='bold 11px sans-serif';ctx.textAlign='center';ctx.fillStyle='#3a2a00';ctx.fillText(txt,p[0],p[1]+4)}}

// ---- the room: four walls, ceiling frame, windows ----
const WALLS=[{a:[-.5,-.5],b:[-.5,H-.5],n:[1,0,0]},{a:[W-.5,-.5],b:[W-.5,H-.5],n:[-1,0,0]},{a:[-.5,-.5],b:[W-.5,-.5],n:[0,0,1]},{a:[-.5,H-.5],b:[W-.5,H-.5],n:[0,0,-1]}];
const wp=(w,t,h)=>{const q=P(w.a[0]+(w.b[0]-w.a[0])*t,h,w.a[1]+(w.b[1]-w.a[1])*t);return q};
function drawWall(w){
  const far=NZ(w.n[0],w.n[1],w.n[2])[1]<0, L=Math.hypot(w.b[0]-w.a[0],w.b[1]-w.a[1]);
  const q=[wp(w,0,0),wp(w,1,0),wp(w,1,WH),wp(w,0,WH)];
  if(!far){ctx.beginPath();ctx.moveTo(q[3][0],q[3][1]);ctx.lineTo(q[2][0],q[2][1]);ctx.strokeStyle=INK+'.35)';ctx.lineWidth=1.3;ctx.stroke();return}
  poly(q,'rgba(255,255,255,.075)',null);
  ctx.strokeStyle=INK+'.17)';ctx.lineWidth=1;ctx.beginPath();
  for(let u=2;u<L;u+=2){const a=wp(w,u/L,0),b=wp(w,u/L,WH);ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1])}
  {const a=wp(w,0,1.1),b=wp(w,1,1.1);ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1])} ctx.stroke();
  for(let c=3;c<L-1;c+=5){const t0=(c-1)/L,t1=(c+1)/L,wq=[wp(w,t0,1.25),wp(w,t1,1.25),wp(w,t1,2.35),wp(w,t0,2.35)];
    poly(wq,'rgba(170,225,255,.20)',INK+'.95)',1.6);
    const m=[wp(w,(t0+t1)/2,1.25),wp(w,(t0+t1)/2,2.35),wp(w,t0,1.8),wp(w,t1,1.8)];
    ctx.beginPath();ctx.moveTo(m[0][0],m[0][1]);ctx.lineTo(m[1][0],m[1][1]);ctx.moveTo(m[2][0],m[2][1]);ctx.lineTo(m[3][0],m[3][1]);ctx.strokeStyle=INK+'.6)';ctx.lineWidth=1;ctx.stroke()}
  poly(q,null,INK+'.9)',1.9)}

function robot(gx,gy){
  const st='rgba(255,255,255,.95)', k=1.7;
  wbox(gx-.21*k,gy,.07*k,.27*k,0,.2*k,st,'rgba(20,30,60,.7)',1.5); wbox(gx+.21*k,gy,.07*k,.27*k,0,.2*k,st,'rgba(20,30,60,.7)',1.5);
  wbox(gx,gy,.17*k,.22*k,.2*k,.62*k,st,'rgba(255,196,0,.85)',1.6);
  line3([gx,.62*k,gy],[gx,.78*k,gy],st,2);
  wbox(gx,gy,.21*k,.1*k,.78*k,.95*k,st,'rgba(255,196,0,.75)',1.6);
  for(const dx of [-.11*k,.11*k]){const e=P(gx+dx,.87*k,gy);ctx.beginPath();ctx.arc(e[0],e[1],4,0,6.3);ctx.fillStyle='#bff6ff';ctx.fill();ctx.strokeStyle='#04204f';ctx.lineWidth=1;ctx.stroke()}
  line3([gx,.95*k,gy],[gx,1.9*k,gy],'rgba(255,230,130,.8)',1.2);label('WALL-E',gx,1.9*k+.15,gy,'#ffe9a8',11)}
function egg(gx,gy,now){
  const bob=0.72+0.07*Math.sin(now/380), p=P(gx,bob,gy), s=C.f/(C.d0+p[2]), rx=.28*s, ry=.44*s;
  const sh=P(gx,0.02,gy);ctx.beginPath();ctx.ellipse(sh[0],sh[1],rx*.9,rx*.35,0,0,6.3);ctx.fillStyle='rgba(0,20,80,.35)';ctx.fill();
  const g=ctx.createRadialGradient(p[0]-rx*.35,p[1]-ry*.45,rx*.1,p[0],p[1],ry);g.addColorStop(0,'#ffffff');g.addColorStop(1,'#bcd9ff');
  ctx.shadowColor='#d8f1ff';ctx.shadowBlur=22;ctx.beginPath();ctx.ellipse(p[0],p[1],rx,ry,0,0,6.3);ctx.fillStyle=g;ctx.fill();ctx.shadowBlur=0;
  ctx.beginPath();ctx.ellipse(p[0],p[1]-ry*.22,rx*.74,ry*.2,0,0,6.3);ctx.fillStyle='#0b2a66';ctx.fill();
  ctx.fillStyle='#5be7ff';for(const dx of [-.3,.3]){ctx.beginPath();ctx.ellipse(p[0]+rx*dx,p[1]-ry*.22,rx*.11,ry*.075,0,0,6.3);ctx.fill()}
  ctx.font='bold 12px Segoe UI,sans-serif';ctx.textAlign='center';ctx.fillStyle='#ffe9a8';ctx.fillText('EVE',p[0],p[1]-ry-8)}

function draw(now){
  if(auto)yaw+=0.004;
  C=mk(); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,pw,ph);
  const f=F[fi], bel=D.beliefs[f.b], cur=f.pos;
  // floor + grid
  poly([P(-.5,0,-.5),P(W-.5,0,-.5),P(W-.5,0,H-.5),P(-.5,0,H-.5)],'rgba(255,255,255,.055)',INK+'.85)',1.8);
  ctx.beginPath();for(let i=0;i<=W;i++){const a=P(i-.5,0,-.5),b=P(i-.5,0,H-.5);ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1])}
  for(let j=0;j<=H;j++){const a=P(-.5,0,j-.5),b=P(W-.5,0,j-.5);ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1])}
  ctx.strokeStyle=INK+'.17)';ctx.lineWidth=1;ctx.stroke();
  // belief = glowing floor heat
  if(layers.belief)for(let y=0;y<H;y++)for(let x=0;x<W;x++){const t=bel[y][x];if(t<0.03||wall.has(x+','+y))continue;
    poly([P(x-.5,0.01,y-.5),P(x+.5,0.01,y-.5),P(x+.5,0.01,y+.5),P(x-.5,0.01,y+.5)],rgba(ramp(t),Math.min(.9,.14+.9*t)),null)}
  // base pad
  {const s0=D.start;poly([P(s0[0]-.45,.02,s0[1]-.45),P(s0[0]+.45,.02,s0[1]-.45),P(s0[0]+.45,.02,s0[1]+.45),P(s0[0]-.45,.02,s0[1]+.45)],'rgba(91,231,255,.30)','#9ff1ff',2);label('BASE',s0[0],.05,s0[1]+.9,'#d6f7ff',10)}
  // far walls first (we look through the near ones: cutaway)
  WALLS.forEach(w=>{if(NZ(w.n[0],w.n[1],w.n[2])[1]<0)drawWall(w)});
  // ceiling beams (faint)
  ctx.beginPath();for(let u=2.5;u<W;u+=5){const a=P(u-.5,WH,-.5),b=P(u-.5,WH,H-.5);ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1])}
  for(let v=2.5;v<H;v+=5){const a=P(-.5,WH,v-.5),b=P(W-.5,WH,v-.5);ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1])}ctx.strokeStyle=INK+'.10)';ctx.lineWidth=1;ctx.stroke();
  // sorted room contents
  const obj=[];
  for(const w of D.walls)obj.push({z:P(w[0],0,w[1])[2],fn:()=>{wbox(w[0],w[1],.45,.45,0,1.0,AMB+'.95)',AMB+'.16)',1.5);
      line3([w[0]-.45,1,w[1]-.45],[w[0]+.45,1,w[1]+.45],AMB+'.5)',1);line3([w[0]+.45,1,w[1]-.45],[w[0]-.45,1,w[1]+.45],AMB+'.5)',1)}});
  if(layers.belief)for(let y=0;y<H;y++)for(let x=0;x<W;x++){const t=bel[y][x];if(t<0.4||wall.has(x+','+y))continue;
    obj.push({z:P(x,0,y)[2]+0.001,fn:()=>{const c=ramp(t);wbox(x,y,.36,.36,0,.15+1.7*t,rgba(c,.95),rgba(c,.20),1.2)}})}
  const eveShown=layers.truth||(found&&fi===last);
  if(eveShown)obj.push({z:P(D.eve[0],0,D.eve[1])[2]+0.002,fn:()=>egg(D.eve[0],D.eve[1],now)});
  if(layers.clues){const rv=revealed[fi];for(const c of D.clues){const seen=rv.has(c.id);if(!seen&&!layers.truth)continue;
    obj.push({z:P(c.pos[0],0,c.pos[1])[2]+0.002,fn:()=>{const col=seen?'#ffd24a':(c.reliable?'#7dffb0':'#ff8a8a');
      line3([c.pos[0],0.05,c.pos[1]],[c.pos[0],1.0,c.pos[1]],col,1.2);
      if(seen)diamond(c.pos[0],1.2,c.pos[1],9,'#ffd24a','#fff','?');else diamond(c.pos[0],1.2,c.pos[1],8,'rgba(0,0,0,0)',col)}})}}
  obj.sort((a,b)=>b.z-a.z);obj.forEach(o=>o.fn());robot(cur[0],cur[1]);
  // clue pointers
  if(layers.clues){const rv=revealed[fi];ctx.setLineDash([5,4]);for(const id of rv){const c=clueById[id];const a=P(c.pos[0],.25,c.pos[1]),b=P(c.estimate[0],.25,c.estimate[1]);
    ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.strokeStyle=layers.truth?(c.reliable?'#7dffb0':'#ff8a8a'):'#ffd24a';ctx.lineWidth=1.3;ctx.stroke();
    ctx.beginPath();ctx.arc(b[0],b[1],4,0,6.3);ctx.stroke()}ctx.setLineDash([])}
  // trail
  if(layers.trail){ctx.beginPath();let first=true;for(let i=0;i<=fi;i++){const p=P(F[i].pos[0],.06,F[i].pos[1]);first?ctx.moveTo(p[0],p[1]):ctx.lineTo(p[0],p[1]);first=false}
    ctx.strokeStyle='rgba(190,245,255,.95)';ctx.lineWidth=3;ctx.shadowColor='#7fe3ff';ctx.shadowBlur=10;ctx.stroke();ctx.shadowBlur=0;
    for(let i=0;i<=fi;i++)if(F[i].type==='scan'){const p=P(F[i].pos[0],.06,F[i].pos[1]);ctx.beginPath();ctx.arc(p[0],p[1],5,0,6.3);ctx.fillStyle='#ffb347';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke()}}
  // "target lock" beam on the current best guess
  {let bt=0,bx=0,by=0;for(let y=0;y<H;y++)for(let x=0;x<W;x++)if(bel[y][x]>bt){bt=bel[y][x];bx=x;by=y}
   if(!eveShown&&f.conf>=0.12){ctx.shadowColor='#fff';ctx.shadowBlur=12;line3([bx,0.02,by],[bx,WH,by],'rgba(255,255,255,.75)',1.6);ctx.shadowBlur=0;
     const th=.5+.5*Math.sin(now/300);ring(bx,by,.5+.35*th,'#ffffff',2,.9-.4*th,.05);label('EST. '+Math.round(f.conf*100)+'%',bx,WH+.25,by,'#fff',11)}}
  // scan ping
  if(f.type==='scan'){const t=(now-frameStart)/1100;if(t<1.3){const rd=Math.min(1,Math.max(0,f.reading||0)),base=.4+.6*rd;
    for(const lag of [0,.14,.28]){const u=t-lag;if(u>0&&u<1)ring(cur[0],cur[1],u*D.scanRange,'#d8f6ff',3.2-2*u,(1-u)*base)}}}
  if(eveShown&&found&&fi===last){const th=Math.abs(Math.sin(now/260));[.8,1.4,2.1].forEach((r,i)=>ring(D.eve[0],D.eve[1],r+th*(.4+.25*i),'#ffd24a',2.4,.95-.22*i,.06))}
  // near walls: just their top edges, plus corner posts, so the room still reads as a room
  WALLS.forEach(w=>{if(NZ(w.n[0],w.n[1],w.n[2])[1]>=0)drawWall(w)});
  for(const [x,z] of [[-.5,-.5],[W-.5,-.5],[W-.5,H-.5],[-.5,H-.5]])line3([x,0,z],[x,WH,z],INK+'.55)',1.6);
  const bb=document.getElementById('badge');if(fi===last){bb.style.display='block';bb.textContent=D.status+' at ('+D.inferred.join(', ')+')'}else bb.style.display='none';
}
function hud(){const f=F[fi];document.getElementById('cnt').textContent='frame '+fi+' / '+last;scrub.value=fi;
  document.getElementById('st').innerHTML='step <b>'+f.step+'</b> &nbsp; battery <b>'+f.battery+'</b>/'+D.battFull+'<span id="bb"><i style="width:'+Math.round(100*f.battery/D.battFull)+'%"></i></span><br>confidence <b>'+Math.round(f.conf*100)+'%</b> &nbsp; time left <b>'+f.time+'</b>'+(f.reading!==null?'<br>signal return <b>'+f.reading.toFixed(2)+'</b>':'');
  document.getElementById('why').innerHTML='<span class="tag">'+f.decision.toUpperCase()+'</span>'+f.reasoning.replace(/&/g,'&amp;').replace(/</g,'&lt;');
  document.getElementById('pl').textContent=playing?'Pause':(fi===last?'Replay':'Play')}
function setFrame(n){fi=Math.max(0,Math.min(last,n));frameStart=lastAdv=performance.now();hud()}
function loop(now){ if(playing){const f=F[fi],d=(f.type==='scan'?1300:f.decision==='replan'?520:f.type==='declare'?800:240)/speed;
    if(now-lastAdv>d){if(fi<last)setFrame(fi+1);else{playing=false;hud()}}}
  draw(now);requestAnimationFrame(loop)}
document.querySelectorAll('[data-l]').forEach(e=>e.onclick=()=>{layers[e.dataset.l]=!layers[e.dataset.l];e.classList.toggle('on',layers[e.dataset.l])});
document.querySelectorAll('[data-v]').forEach(e=>e.onclick=()=>{const v=e.dataset.v;auto=false;document.getElementById('auto').classList.remove('on');
  if(v==='iso'){yaw=.8;pitch=.82}if(v==='top'){yaw=0;pitch=1.5}if(v==='side'){yaw=.4;pitch=.4}zoom=1});
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
