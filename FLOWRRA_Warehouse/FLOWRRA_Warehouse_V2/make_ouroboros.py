import math, json, html
CX=CY=500.0
R_CENTER=118; R_TRACK=129
RINGS=[("step","The step",140,205),("data","Data flow",209,274),("mech","Mechanism",278,343),
       ("carry","Carries over",347,412),("team","Team",416,478)]
RING_MEANING={"step":"What happens in this phase.","data":"What goes in, and what comes out.",
 "mech":"How it is built.","carry":"What persists into the next cycle.","team":"Who would own it."}
C=lambda s:f"<code>{s}</code>"
PH=[
 dict(t="Repair",col="#E63946",
  arcs={"step":["0. Repair"],"data":["conflicts in","repaired world out"],"mech":["Three-tier collapse","escape, rewind, yield"],
        "carry":["Yield holds","and pair history"],"team":["Operations research & recovery"]},
  step="Fixes the conflicts the previous step's moves created, before any fleet looks.",
  din=f"{C('proximity')}, {C('_step_deadlocked')}, {C('_step_warning')}",
  dout=f"repaired {C('current_pos')}, {C('yield_durations')}, a rebuilt proximity index and every fleet's {C('sf_peer_proximity')}",
  mech="Wave Function Collapse in three tiers: spatial escape, temporal rewind, right-of-way yield. It escalates when the same pair keeps meeting, by repeat collision or by recurrence within 2 cells; convoys are exempt. The recovery head invokes it as one holon-level decision, or it is forced on deadlock. If it moves anyone, the snapshot is re-taken for every fleet.",
  carry=f"Yield holds ({C('_yield_until')}), each pair's collision and recurrence count, and position history for temporal rewinds.",
  team="Operations research &amp; recovery: the collapse protocol, escalation rules and recovery policy."),
 dict(t="Look",col="#F4A261",
  arcs={"step":["1. Look"],"data":["777 numbers","per fleet"],"mech":["84 base + 3 × 231","density diamond"],
        "carry":["Slow density memory","half-life about 53 steps"],"team":["Physics & perception"]},
  step="Every fleet gets its own view of the same moment: its neighbourhood and its goal, plus one warehouse-wide number, the holon's integrity.",
  din=f"repaired {C('current_pos')}, graph {C('G')}, the density field, {C('aisle_index')}",
  dout=f"{C('node_features')} of 777 per fleet; adjacency with 4 edge channels (closeness, conflict, head-on, arrival order)",
  mech="84 base features: goal gradient, rays, situation flags, and the holon's integrity (0, 0.5 or 1), recomputed fresh before every look. 693 density features: a 231-cell diamond in three channels, mask, repulsion and slow memory.",
  carry="The density field remembers. Its fast memory fades within about 2 steps; the slow channel has a half-life of about 53 steps, long enough to learn that a corridor keeps jamming.",
  team="Physics &amp; perception: the density field, rays and the state layout."),
 dict(t="Decide",col="#E9C46A",
  arcs={"step":["2. Decide"],"data":["one action per fleet","one recovery mode"],"mech":["Gated 3D conv + attention","heads weighted 6 : 4 : 1"],
        "carry":["Exploration schedule","low, peak, low"],"team":["Multi-agent policy"]},
  step="One shared policy picks an action for every fleet, all from that same moment.",
  din=f"{C('node_features')}, adjacency, {C('valid_action_masks')}",
  dout=f"{C('actions')} 0 to 6 per fleet, {C('action_ranking')}, and one recovery mode for the holon ({C('last_recovery_q')})",
  mech="A gated 3D convolution reads the density diamond; graph attention lets neighbours inform each other. Three value heads, safety, delivery and efficiency, are combined 6 : 4 : 1 to choose, so safety outranks delivery and delivery outranks efficiency. A separate recovery head makes the holon's recovery decision. Action masks veto impossible moves; epsilon-greedy explores.",
  carry="Epsilon follows a gaussian over the run: near-greedy at the start, most exploratory mid-run, near-greedy again at the end.",
  team="Multi-agent policy: the network, action masks and exploration."),
 dict(t="Move",col="#2A9D8F",
  arcs={"step":["3. Move"],"data":["positions, claims","and waits"],"mech":["Everyone moves at once,","then one frozen snapshot"],
        "carry":["Goal claims","and open pickups"],"team":["Fleet operations"]},
  step="All fleets move at once. Consequences are then settled against one frozen snapshot, so no fleet gains by being processed first.",
  din=f"{C('actions')}, {C('valid_action_masks')}, tabu actions, proximity for braking",
  dout=f"new positions, {C('waiting_nodes')}, goal claims, stopped fleets",
  mech="Phase A: every fleet moves. Snapshot. Phase B: waits, goal claims and arrivals are judged against it. Proximity braking, tabu, and a BFS fallback for livelock. VDA 5050-style hardware halts, with rescue handovers for dead fleets.",
  carry="Goal claims, waiting timers, and dead fleets with their open pickups until a rescuer arrives.",
  team="Fleet operations: movement, braking, fault halts and rescue handovers."),
 dict(t="Judge",col="#457B9D",
  arcs={"step":["4. Judge"],"data":["rewards per fleet","3 heads + holon"],"mech":["Pay for your own part","fixed value scales"],
        "carry":["Nothing:","judged fresh each step"],"team":["Reward design"]},
  step="Each fleet is charged for what its own move caused. The holon's own decision, recovery, is judged by the holon.",
  din=f"{C('old_dist')} against {C('new_dist')}, claimed goals, post-action conflicts",
  dout="rewards of 4 columns per fleet: safety, delivery, efficiency, and the holon's integrity for the recovery head",
  mech="Each head is divided by a fixed value scale, calibrated on cold_run18: safety 34.3, delivery 70.4, efficiency 17.5. Collisions are charged to the move that caused them; integrity and recovery costs go only to the fleets involved. The recovery head gets the holon's integrity and every recovery event at full size. No baseline. The 6 : 4 : 1 priority is applied when choosing, in Decide.",
  carry="Nothing: every step is judged fresh.",
  team="Reward design: the priority heads, attribution and calibration."),
 dict(t="Remember & learn",col="#9C89B8",
  arcs={"step":["5. Remember","and learn"],"data":["64 past steps","per update"],"mech":["Double DQN","online picks, target scores"],
        "carry":["Buffer of 30,000,","target net every 1,000"],"team":["MLOps"]},
  step="The whole step goes into the replay buffer, then one learning step, drawn from the past as well as the present.",
  din="states, adjacency, actions, rewards, next states, next valid masks, the active mask, done",
  dout=f"updated {C('policy_net')} weights; {C('target_net')} copied every 1,000 learning steps",
  mech="Double DQN: the online network chooses the next action and the target network scores it, which reduces plain DQN's upward bias. Impossible next actions are masked. Huber loss per head, over active fleets only; gradients clipped at 1.0.",
  carry="The replay buffer, 30,000 transitions or about 38 episodes, and the target network, a hard copy every 1,000 learning steps, about every 1.3 episodes.",
  team="MLOps: the replay buffer, Double DQN, target sync and training stability."),
]
def pol(r,a): a=math.radians(a); return (CX+r*math.sin(a), CY-r*math.cos(a))
def sector(r0,r1,a0,a1):
    big=1 if a1-a0>180 else 0
    p=pol(r1,a0);q=pol(r1,a1);s=pol(r0,a1);t=pol(r0,a0)
    return f"M{p[0]:.2f},{p[1]:.2f} A{r1},{r1} 0 {big} 1 {q[0]:.2f},{q[1]:.2f} L{s[0]:.2f},{s[1]:.2f} A{r0},{r0} 0 {big} 0 {t[0]:.2f},{t[1]:.2f} Z"
def arcpath(r,a0,a1,cw):
    if cw: p,q,sw=pol(r,a0),pol(r,a1),1
    else:  p,q,sw=pol(r,a1),pol(r,a0),0
    return f"M{p[0]:.2f},{p[1]:.2f} A{r:.2f},{r:.2f} 0 0 {sw} {q[0]:.2f},{q[1]:.2f}"
def hexrgb(h): h=h.lstrip('#'); return [int(h[i:i+2],16) for i in (0,2,4)]
def mix(c,bg,m): a,b=hexrgb(c),hexrgb(bg); return "#%02x%02x%02x"%tuple(round(m*x+(1-m)*y) for x,y in zip(a,b))
def lum(h):
    v=[x/255 for x in hexrgb(h)]; v=[x/12.92 if x<=0.03928 else ((x+0.055)/1.055)**2.4 for x in v]
    return 0.2126*v[0]+0.7152*v[1]+0.0722*v[2]
def contrast(a,b): la,lb=sorted([lum(a),lum(b)],reverse=True); return (la+0.05)/(lb+0.05)
THEMES={"dark":dict(bg="#12141c",ink="#e8eaf0"),"light":dict(bg="#eef1f5",ink="#1b1e27")}
MIXK=[0.62,0.72,0.82,0.90,0.97]; WHITE="#ffffff"; DARKINK="#16181f"
def colors(theme):
    bg=THEMES[theme]["bg"]; out={}
    for i,p in enumerate(PH):
        for k,(rk,_,_,_) in enumerate(RINGS):
            f=mix(p["col"],bg,MIXK[k]); t=WHITE if contrast(WHITE,f)>=contrast(DARKINK,f) else DARKINK
            out[(i,k)]=(f,t,max(contrast(WHITE,f),contrast(DARKINK,f)))
    return out
FS={"step":15.5,"data":13,"mech":13,"carry":13,"team":13.5}
def wheel(theme=None, inline=False, font="Barlow Semi Condensed"):
    col=colors(theme) if inline else None
    o=[]; defs=[]
    o.append(f'<circle cx="{CX}" cy="{CY}" r="{R_CENTER}" class="hub"' + (f' fill="{"#181b24" if theme=="dark" else "#ffffff"}"' if inline else '') + '/>')
    o.append(f'<circle cx="{CX}" cy="{CY}" r="{R_TRACK}" class="track" fill="none"' + (f' stroke="{"#3a4054" if theme=="dark" else "#c3c9d4"}" stroke-width="1"' if inline else '') + '/>')
    for i,p in enumerate(PH):
        a0=i*60-30+1.4; a1=i*60+30-1.4; mid=i*60; top=math.cos(math.radians(mid))>0
        for k,(rk,rn,r0,r1) in enumerate(RINGS):
            fill_attr=f' fill="{col[(i,k)][0]}"' if inline else ''
            label=f'{p["t"]}, {rn}'
            o.append(f'<path d="{sector(r0,r1,a0,a1)}" class="arc a{i}{k}" data-phase="{i}" data-ring="{rk}" tabindex="0" role="button" aria-label="{html.escape(label)}"{fill_attr}/>')
            lines=p["arcs"][rk]; fs=FS[rk]; rm=(r0+r1)/2; d=0.62*fs
            offs=[0.0] if len(lines)==1 else ([d,-d] if top else [-d,d])
            for j,(txt,off) in enumerate(zip(lines,offs)):
                rl=rm+off; rb=rl-0.35*fs if top else rl+0.35*fs
                pid=f"tp{i}{k}{j}"
                defs.append(f'<path id="{pid}" d="{arcpath(rb,a0+2,a1-2,top)}" fill="none"/>')
                tf=f' fill="{col[(i,k)][1]}"' if inline else ''
                weight=700 if rk=="step" else (600 if j==0 else 500)
                o.append(f'<text class="al t{i}{k}{" al-step" if rk=="step" else ""}" font-size="{fs}" font-weight="{weight}"{tf}'
                         + (f' font-family="{font}"' if inline else '') +
                         f'><textPath href="#{pid}" xlink:href="#{pid}" startOffset="50%" text-anchor="middle">{html.escape(txt)}</textPath></text>')
    # clockwise chevrons at phase boundaries, on the track
    for i in range(6):
        b=i*60+30; px,py=pol(R_TRACK,b); a=math.radians(b)
        dx,dy=math.cos(a),math.sin(a); nx,ny=math.sin(a),-math.cos(a)
        tip=(px+6*dx,py+6*dy); l=(px-4*dx+5*nx,py-4*dy+5*ny); r=(px-4*dx-5*nx,py-4*dy-5*ny)
        o.append(f'<path class="chev" d="M{l[0]:.2f},{l[1]:.2f} L{tip[0]:.2f},{tip[1]:.2f} L{r[0]:.2f},{r[1]:.2f}" fill="none"'
                 + (f' stroke="{"#8f97ad" if theme=="dark" else "#5a6274"}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"' if inline else '') + '/>')
    # Learn -> Decide feedback, bowing below the title
    P1=pol(104,272); P2=pol(104,148); Cc=(470,622)
    fb=f"M{P1[0]:.2f},{P1[1]:.2f} Q{Cc[0]:.2f},{Cc[1]:.2f} {P2[0]:.2f},{P2[1]:.2f}"
    o.append(f'<path class="fb" d="{fb}" fill="none"' + (f' stroke="#9C89B8" stroke-width="2" stroke-dasharray="5 5"' if inline else '') + '/>')
    # arrowhead at P2 along the curve tangent (Q-curve end tangent = P2 - Cc)
    tx,ty=P2[0]-Cc[0],P2[1]-Cc[1]; n=math.hypot(tx,ty); tx,ty=tx/n,ty/n; nx,ny=-ty,tx
    h=(P2[0]-8*tx+5*nx,P2[1]-8*ty+5*ny); g=(P2[0]-8*tx-5*nx,P2[1]-8*ty-5*ny)
    o.append(f'<path class="fbhead" d="M{h[0]:.2f},{h[1]:.2f} L{P2[0]:.2f},{P2[1]:.2f} L{g[0]:.2f},{g[1]:.2f}" fill="none"'
             + (' stroke="#9C89B8" stroke-width="2" stroke-linecap="round"' if inline else '') + '/>')
    ink=THEMES[theme]["ink"] if inline else None
    o.append(f'<text x="{CX}" y="{CY-14}" class="hub-title" text-anchor="middle"' + (f' fill="{ink}" font-family="{font}" font-size="34" font-weight="700" letter-spacing="3"' if inline else '') + '>FLOWRRA</text>')
    o.append(f'<text x="{CX}" y="{CY+12}" class="hub-sub" text-anchor="middle"' + (f' fill="#8f97ad" font-family="{font}" font-size="14"' if inline else '') + '>one step, six phases</text>')
    o.append(f'<text x="476" y="566" class="fb-label" text-anchor="middle"' + (f' fill="#9C89B8" font-family="{font}" font-size="12.5"' if inline else '') + '>new weights</text>')
    return "<defs>"+"".join(defs)+"</defs>"+"".join(o)

# ---------------------------------------------------------------------------
# PAGE. Everything above is the content and geometry; edit text in PH, rings
# in RINGS (radii are in the 1000x1000 viewBox), then run:
#   python3 make_ouroboros.py          -> writes flowrra_architecture.html
#   python3 make_ouroboros.py --qa     -> also writes qa_dark.png (pip install cairosvg)
# ---------------------------------------------------------------------------
import json, sys
def varblock(theme):
    c=colors(theme); return "".join(f"--a{i}{k}:{c[(i,k)][0]};--t{i}{k}:{c[(i,k)][1]};" for (i,k) in c)
def page():
    svg_body=wheel(theme=None,inline=False)
    classes="".join(f".a{i}{k}{{fill:var(--a{i}{k})}}.t{i}{k}{{fill:var(--t{i}{k})}}" for i in range(len(PH)) for k in range(len(RINGS)))
    phases=[dict(n=f"{i}. {p['t']}",col=p["col"],step=p["step"],din=p["din"],dout=p["dout"],mech=p["mech"],carry=p["carry"],team=p["team"]) for i,p in enumerate(PH)]
    rings=[dict(k=k,name=n,meaning=RING_MEANING[k]) for k,n,_,_ in RINGS]
    data=json.dumps(dict(phases=phases,rings=rings)).replace("</","<\\/")
    return TEMPLATE.format(LIGHT=LIGHT+varblock("light"),DARK=DARK+varblock("dark"),classes=classes,svg_body=svg_body,data=data)
DARK="--bg:#12141c;--panel:#181b24;--line:#2a2f3f;--ink:#e8eaf0;--muted:#98a0b5;--code-bg:#0f1118;--code-ink:#a9dcb3;--hub:#181b24;--track:#3a4054;--chev:#8f97ad;"
LIGHT="--bg:#eef1f5;--panel:#ffffff;--line:#d6dbe4;--ink:#1b1e27;--muted:#586072;--code-bg:#eef1f5;--code-ink:#2c6a3b;--hub:#ffffff;--track:#c3c9d4;--chev:#5a6274;"

TEMPLATE = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n<title>FLOWRRA Ouroboros</title>\n<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n<link href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">\n<style>\n:root{{{LIGHT}box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}\n@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{{DARK}}}}}\n:root[data-theme="dark"]{{{DARK}}}\nhtml{{scroll-padding-top:env(safe-area-inset-top,0px);height:100%}}\n*,*::before,*::after{{box-sizing:inherit}}\nbody{{margin:0;min-height:100%;background:var(--bg);color:var(--ink);font-family:Barlow,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:16px;line-height:1.55;\n  display:grid;grid-template-columns:minmax(0,1fr) minmax(320px,440px)}}\n.wheel{{display:flex;align-items:center;justify-content:center;padding:clamp(12px,3vw,32px);min-width:0}}\n.wheel svg{{width:100%;max-width:min(92vh,960px);height:auto;display:block}}\n.panel{{background:var(--panel);border-left:1px solid var(--line);padding:32px 30px 40px;overflow-y:auto;max-height:100vh;position:sticky;top:0}}\nh1{{font-family:"Barlow Semi Condensed","Arial Narrow",sans-serif;font-weight:700;font-size:30px;line-height:1.15;margin:0 0 12px;padding-left:12px;border-left:5px solid var(--muted)}}\n.intro{{color:var(--muted);margin:0 0 22px;max-width:62ch}}\n.rings{{list-style:none;margin:0;padding:0}}\n.rings li{{padding:12px 14px;margin:0 0 6px;border-left:3px solid transparent;border-radius:0 6px 6px 0}}\n.rings li.on{{background:color-mix(in srgb,var(--pc,#888) 12%,transparent);border-left-color:var(--pc,#888)}}\n.rings h2{{font-family:"Barlow Semi Condensed","Arial Narrow",sans-serif;font-weight:600;font-size:15px;letter-spacing:.02em;margin:0 0 4px;color:var(--muted)}}\n.rings li.on h2{{color:var(--ink)}}\n.rb p{{margin:0 0 6px;max-width:62ch}}\n.k{{font-weight:600;margin-right:6px}}\ncode{{font-family:"JetBrains Mono",Consolas,monospace;font-size:.84em;background:var(--code-bg);color:var(--code-ink);padding:1px 5px;border-radius:4px;word-break:break-word}}\n.hub{{fill:var(--hub);stroke:var(--line);stroke-width:2}}\n.track{{stroke:var(--track);stroke-width:1}}\n.chev{{stroke:var(--chev);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}}\n.fb,.fbhead{{stroke:#9C89B8;stroke-width:2;stroke-linecap:round}}.fb{{stroke-dasharray:5 5}}\n.hub-title{{fill:var(--ink);font-family:"Barlow Semi Condensed","Arial Narrow",sans-serif;font-size:34px;font-weight:700;letter-spacing:3px}}\n.hub-sub{{fill:var(--muted);font-family:Barlow,sans-serif;font-size:14px}}\n.fb-label{{fill:#9C89B8;font-family:Barlow,sans-serif;font-size:13px}}\n.al{{font-family:"Barlow Semi Condensed","Arial Narrow",sans-serif;pointer-events:none}}\n.arc{{stroke:var(--bg);stroke-width:2.5;cursor:pointer;outline:none}}\n.arc:hover{{filter:brightness(1.08)}}\n.arc.sel{{stroke:var(--ink);stroke-width:3}}\n.arc:focus-visible{{stroke:var(--ink);stroke-width:4;stroke-dasharray:6 4}}\n{classes}\n@media (max-width:900px){{body{{grid-template-columns:1fr}}.panel{{border-left:0;border-top:1px solid var(--line);max-height:none;position:static}}.wheel svg{{max-width:100%}}}}\n@media (prefers-reduced-motion:reduce){{*{{transition:none!important}}}}\n</style>\n</head>\n<body>\n<main class="wheel">\n<svg viewBox="0 0 1000 1000" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" role="group" aria-label="FLOWRRA step cycle: six phases, five rings">\n{svg_body}\n</svg>\n</main>\n<aside class="panel" aria-live="polite">\n<h1 id="ptitle">FLOWRRA Ouroboros</h1>\n<p class="intro" id="pintro">One step of the orchestrator, read clockwise from Repair. Each ring zooms further out: what happens, what flows through, how it is built, what carries over to the next cycle, and who would own it. Hover, tap or tab to a segment.</p>\n<ol class="rings" id="rings"></ol>\n</aside>\n<script>\nconst D={data};\nconst ringsEl=document.getElementById(\'rings\');\nD.rings.forEach(r=>{{const li=document.createElement(\'li\');li.dataset.ring=r.k;li.innerHTML=\'<h2>\'+r.name+\'</h2><div class="rb"><p>\'+r.meaning+\'</p></div>\';ringsEl.appendChild(li);}});\nfunction body(p,k){{\n  if(k===\'data\') return \'<p><span class="k">In</span>\'+p.din+\'</p><p><span class="k">Out</span>\'+p.dout+\'</p>\';\n  return \'<p>\'+({{step:p.step,mech:p.mech,carry:p.carry,team:p.team}})[k]+\'</p>\';\n}}\nlet cur=null;\nfunction select(i,ring){{\n  const p=D.phases[i];\n  const t=document.getElementById(\'ptitle\');t.textContent=p.n;t.style.borderLeftColor=p.col;\n  document.getElementById(\'pintro\').textContent=p.step;\n  document.querySelectorAll(\'#rings li\').forEach(li=>{{\n    li.style.setProperty(\'--pc\',p.col);li.querySelector(\'.rb\').innerHTML=body(p,li.dataset.ring);\n    li.classList.toggle(\'on\',li.dataset.ring===ring);}});\n  if(cur!==i){{document.querySelectorAll(\'.arc\').forEach(a=>a.classList.toggle(\'sel\',+a.dataset.phase===i));cur=i;}}\n}}\ndocument.querySelectorAll(\'.arc\').forEach(a=>{{\n  const go=()=>select(+a.dataset.phase,a.dataset.ring);\n  a.addEventListener(\'mouseenter\',go);a.addEventListener(\'focus\',go);a.addEventListener(\'click\',go);\n  a.addEventListener(\'keydown\',e=>{{if(e.key===\'Enter\'||e.key===\' \'){{e.preventDefault();go();}}}});\n}});\n</script>\n</body>\n</html>'

if __name__=="__main__":
    out=page(); open("Flowrra_Architecture.html","w").write(out); print("wrote Flowrra_Architecture.html")
    if "--qa" in sys.argv:
        import cairosvg
        svg=('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1000 1000" width="1000" height="1000">'
             f'<rect width="1000" height="1000" fill="{THEMES["dark"]["bg"]}"/>'+wheel("dark",True,"DejaVu Sans Condensed")+'</svg>')
        cairosvg.svg2png(bytestring=svg.encode(), write_to="qa_dark.png", output_width=1000); print("wrote qa_dark.png")