"""Fixed, outcome-independent synthetic geometry and topology rules."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

STRUCTURES = {0:'A_spiral',1:'B_tree',2:'C_ellipse',3:'bridge_AB',4:'bridge_BC'}
NODES = np.array([[0,-5],[0,-2],[-2,0],[2,0],[-3.8,3],[-.8,3],[.8,3],[3.8,3]],float)
EDGES = [(0,1),(1,2),(1,3),(2,4),(2,5),(3,6),(3,7)]

def generate(seed=20260918):
    rng=np.random.default_rng(seed); rows=[]
    def add(z,structure,branch,progress,theta=None,along=None,length=None,density=None):
        n=len(z)
        frame=pd.DataFrame(dict(z0=z[:,0],z1=z[:,1],structure=structure,branch=branch,
                                progress=progress,theta=np.full(n,np.nan) if theta is None else theta,
                                along=np.full(n,np.nan) if along is None else along,
                                length=np.full(n,np.nan) if length is None else length,
                                analytic_density=np.full(n,np.nan) if density is None else density))
        rows.append(frame)
    # Arclength-uniform ribbon around an Archimedean spiral, 2.5 separated turns.
    dense=np.linspace(-5*np.pi,0,100001); r=1.2+4.4*(dense+5*np.pi)/(5*np.pi)
    speed=np.sqrt(r*r+(4.4/(5*np.pi))**2); arc=np.cumsum(speed);arc=(arc-arc[0])/(arc[-1]-arc[0])
    n=2001; u=(np.arange(n)+rng.uniform(.1,.9,n))/n
    theta=np.interp(u,arc,dense); r=1.2+4.4*(theta+5*np.pi)/(5*np.pi)
    center=np.c_[r*np.cos(theta)-14,r*np.sin(theta)]
    tangent=np.c_[4.4/(5*np.pi)*np.cos(theta)-r*np.sin(theta),4.4/(5*np.pi)*np.sin(theta)+r*np.cos(theta)]
    normal=np.c_[-tangent[:,1],tangent[:,0]];normal/=np.linalg.norm(normal,axis=1)[:,None]
    add(center+rng.uniform(-.18,.18,(n,1))*normal,0,np.floor((theta+5*np.pi)/(2*np.pi)).astype(int),u,theta=theta)
    # Seven finite-width edges; junction neighborhoods are intentionally connected.
    lengths=np.array([np.linalg.norm(NODES[v]-NODES[u]) for u,v in EDGES]); blocks=np.floor(666*lengths/lengths.sum()).astype(int)
    for j in np.argsort(-(666*lengths/lengths.sum()-blocks))[:666-blocks.sum()]: blocks[j]+=1
    for j,((v,w),count,L) in enumerate(zip(EDGES,blocks*3,lengths)):
        u=(np.arange(count)+rng.uniform(.1,.9,count))/count; d=(NODES[w]-NODES[v])/L
        z=NODES[v]+u[:,None]*L*d+rng.uniform(-.18,.18,(count,1))*np.array([-d[1],d[0]])
        add(z,1,j,u,along=u*L,length=L)
    # p(x,y) proportional to exp(1.2 * normalized_x) inside an ellipse.
    accepted=[]
    while sum(len(a) for a in accepted)<2001:
        q=rng.uniform(-1,1,(8000,2)); keep=(np.sum(q*q,axis=1)<=1)&(rng.random(len(q))<np.exp(1.2*(q[:,0]-1)))
        accepted.append(q[keep])
    q=np.concatenate(accepted)[:2001]; add(q*np.array([4.,3.])+[15.,0.],2,0,(q[:,0]+1)/2,density=np.exp(1.2*q[:,0]))
    internal=pd.concat(rows,ignore_index=True); internal['id']=np.arange(len(internal))
    assert len(internal)==6000
    rows=[]
    for structure,vertices in [(3,[[-8.4,0],[-7.,4.5],[-4.8,4.5],[-3.8,3]]),(4,[[3.8,3],[6,5],[9,5],[11,0]])]:
        vertices=np.array(vertices);lengths=np.linalg.norm(np.diff(vertices,axis=0),axis=1);cum=np.r_[0,lengths.cumsum()]
        n=240;distance=(np.arange(n)+.5)/n*cum[-1];segment=np.searchsorted(cum[1:],distance)
        d=np.diff(vertices,axis=0)[segment]/lengths[segment,None]
        z=vertices[segment]+(distance-cum[segment])[:,None]*d+rng.uniform(-.055,.055,(n,1))*np.c_[-d[:,1],d[:,0]]
        add(z,structure,0,distance/cum[-1],along=distance,length=cum[-1])
    bridge=pd.concat(rows,ignore_index=True);bridge['id']=np.arange(6000,6480)
    df=pd.concat([internal,bridge],ignore_index=True)
    # Each local triplet keeps two references. Fixed split across all learning seeds.
    df['split']='reference'; splitrng=np.random.default_rng(seed+1)
    for structure,group in df.groupby('structure',sort=True):
        if structure==1:
            groups=[v for _,v in group.groupby('branch',sort=True)]
        else: groups=[group]
        for g in groups:
            if structure==2:
                # Serpentine spatial ordering provides coverage in the 2D interior.
                xb=np.floor((g.z0.to_numpy()-11)/8*20).astype(int)
                order=np.lexsort((g.z1.to_numpy()*np.where(xb%2,1,-1),xb))
            else: order=np.argsort(g.progress.to_numpy(),kind='stable')
            idx=g.index.to_numpy()[order];assert len(idx)%3==0
            query=idx.reshape(-1,3)[np.arange(len(idx)//3),splitrng.integers(3,size=len(idx)//3)]
            df.loc[query,'split']='query'
    # Linear map stored in float64; float32 is used only by existing learning code.
    A=np.linalg.qr(np.random.default_rng(seed+2).normal(size=(50,2)))[0]
    x=df[['z0','z1']].to_numpy()@A.T
    return df,x,A

def neighbors(y,ref,k=15,self_indices=None):
    d,idx=cKDTree(ref).query(y,k=k+(self_indices is not None))
    if self_indices is not None:
        keep=idx!=np.asarray(self_indices)[:,None]
        # self is present for ordinary inputs; duplicates still handled row by row.
        pairs=[(di[ki][:k],ii[ki][:k]) for di,ii,ki in zip(d,idx,keep)]
        d=np.stack([a for a,b in pairs]);idx=np.stack([b for a,b in pairs])
    return idx,d

def shortcut_flags(source,target,idx,rz):
    """Directed kNN false shortcuts: nonlocal in z + inappropriate winding/branch."""
    s=source.structure.to_numpy()[:,None];t=target.structure.to_numpy()[idx]
    zs=source[['z0','z1']].to_numpy();zt=target[['z0','z1']].to_numpy()[idx]
    nonlocal_z=np.linalg.norm(zs[:,None]-zt,axis=2)>2*np.asarray(rz)[:,None]
    turn=(s==0)&(t==0)&(np.abs(source.theta.to_numpy()[:,None]-target.theta.to_numpy()[idx])>np.pi)
    bs=source.branch.to_numpy()[:,None];bt=target.branch.to_numpy()[idx]
    branch=(s==1)&(t==1)&(bs!=bt)
    # Adjacent edges within 0.6 of the shared junction are legitimate.
    junction=np.zeros_like(branch)
    for a,(u,v) in enumerate(EDGES):
        for b,(w,h) in enumerate(EDGES):
            common=set((u,v))&set((w,h))
            if a==b or not common: continue
            node=next(iter(common))
            da=source.along.to_numpy() if node==u else source.length.to_numpy()-source.along.to_numpy()
            db=target.along.to_numpy() if node==w else target.length.to_numpy()-target.along.to_numpy()
            junction|=(bs==a)&(bt==b)&(da[:,None]<=.6)&(db[idx]<=.6)
    return (turn&nonlocal_z),(branch&~junction&nonlocal_z)

def graph_audit(frame,x,k=15):
    idx,dist=neighbors(x,x,k,np.arange(len(x))); rz=neighbors(frame[['z0','z1']].to_numpy(),frame[['z0','z1']].to_numpy(),15,np.arange(len(x)))[1][:,-1]
    turns,branches=shortcut_flags(frame,frame,idx,rz)
    graph=csr_matrix((np.ones(idx.size),(np.repeat(np.arange(len(x)),k),idx.ravel())),shape=(len(x),len(x)))
    nc,labels=connected_components(graph,directed=False)
    s=frame.structure.to_numpy();cross=[]
    for a in np.unique(s):
        for b in np.unique(s):
            count=int(((s[:,None]==a)&(s[idx]==b)).sum())
            if a!=b and count: cross.append(dict(source=STRUCTURES[a],target=STRUCTURES[b],directed_edges=count))
    allowed={(0,3),(1,3),(1,4),(2,4)}
    unexpected=sum(q['directed_edges'] for q in cross if tuple(sorted((next(k for k,v in STRUCTURES.items() if v==q['source']),next(k for k,v in STRUCTURES.items() if v==q['target'])))) not in allowed)
    # Removing bridge vertices must recover exactly the three internal components.
    internal=np.flatnonzero(s<3);ni,_=connected_components(graph[internal][:,internal],directed=False)
    report=dict(k=k,n=len(x),components=int(nc),component_sizes=np.bincount(labels).tolist(),components_without_bridge=int(ni),
                cross_structure_edges=cross,unexpected_cross_structure_edges=unexpected,wrong_turn_edges=int(turns.sum()),wrong_branch_edges=int(branches.sum()),
                maximum_knn_radius=float(dist[:,-1].max()),bridge_knn_radius_max=float(dist[s>=3,-1].max()) if np.any(s>=3) else None)
    return report,idx,dist,graph
