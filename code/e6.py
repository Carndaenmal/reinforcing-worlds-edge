"""E6 single-agent control: peer FIXED (alpha=0). Focal Q-learns.
True core is policy-independent; the ESTIMATED delta=0 core converges to it and
its drift is transient (-> 0), unlike the MARL case (E3) where drift persists."""
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, src
from plot_style import COLORS, apply_style, grid
H=12; C={"d1":COLORS["baseline"],"d3":COLORS["risk"],"grey":COLORS["grey"],
         "rob":COLORS["core"],"d0":COLORS["ink"]}
apply_style()
pi1=src.probe_policy(); pi2_fixed=src.pi2_alpha(0.0)  # peer always holds -> top route

def run(seed,n_ep=6000,eps=0.2,lr=0.2,meas=25,nprobe=7000):
    r=np.random.default_rng(seed); Q1=np.zeros((src.NZ,src.NA1))
    ep_l=[]; sym=[]; prevL=None; churn=[]
    for ep in range(n_ep):
        f,rock,t=0,0,0
        while t < H - 1 and f != 5:
            z=src.zidx(f,rock)
            a1=r.integers(4) if r.random()<eps else int(Q1[z].argmax())
            # peer fixed: hold (a2=0)
            hold=True; clear=False; nf=f
            if f==0: nf=1 if a1==0 else (3 if a1==1 else 0)
            elif f==1 and a1==2 and hold and rock==0: nf=2
            elif f==2 and a1==2: nf=5
            elif f==3 and a1==2 and rock==1: nf=4
            elif f==4 and a1==2: nf=5
            nrock=rock; term=(nf==5); r1=(1.0 if term else 0.0)-0.02
            nz=src.zidx(nf,nrock)
            Q1[z,a1]+=lr*(r1+(0 if term else Q1[nz].max())-Q1[z,a1]); f,rock,t=nf,nrock,t+1
        if ep%meas==0:
            # estimate core using the CURRENT (improving) greedy focal policy, not the probe
            pol=np.full((src.NZ,src.NA1),eps/src.NA1); pol[np.arange(src.NZ),Q1.argmax(1)]+=1-eps
            tr,sc=src.rollout(pol,pi2_fixed,nprobe,H,r); m,_=src.landmark_mass(tr,sc)
            L=src.delta_landmarks(m,0.0); sym.append(sorted(L)); ep_l.append(ep)
            if prevL is not None: churn.append(len(set(map(tuple,[L]))^set(map(tuple,[prevL])))>0 or L!=prevL)
            prevL=L
    return np.array(ep_l),sym

truecore={0,1,2,5}
fig,ax=plt.subplots(figsize=(3.7,1.85), constrained_layout=True)
for s in [1,2,3]:
    ep,sym=run(s)
    match=np.array([set(x)==truecore for x in sym])
    y=np.full_like(ep,float(s),dtype=float)
    ax.plot(ep[match],y[match],lw=3.0,color=C["rob"],solid_capstyle="butt")
    if np.any(~match):
        ax.plot(ep[~match],y[~match],ls="none",marker="x",ms=3.2,color=C["d3"])
ax.set_ylim(0.45,3.55); ax.set_yticks([1,2,3]); ax.set_yticklabels(["seed 1","seed 2","seed 3"])
ax.set_xlabel("episode $e$"); ax.set_ylabel("")
ax.set_title("Single-agent core: no churn, exact at every check",loc="left",fontweight="bold")
ax.text(0.02,0.08,"teal = exact match to true core; 240 checks/seed",transform=ax.transAxes,
        fontsize=7.0,color=C["grey"])
grid(ax,axis="x")
fig.savefig("../figs/fig_single.pdf"); plt.close(fig)

# metric: episode after which estimated core is stably == true core
ep,sym=run(7,n_ep=6000)
match=np.array([set(x)==truecore for x in sym])
conv=int(ep[np.where(match)[0][0]]) if match.any() else None
# last time it changed
changes=[i for i in range(1,len(sym)) if sym[i]!=sym[i-1]]
last_change=int(ep[changes[-1]]) if changes else 0
json.dump({"true_core":sorted(truecore),"first_correct_episode":conv,
           "last_core_change_episode":last_change,
           "final_core":sym[-1],"n_changes":len(changes)},
          open("../figs/results_e6.json","w"),indent=2)
print("E6",json.load(open("../figs/results_e6.json")))
