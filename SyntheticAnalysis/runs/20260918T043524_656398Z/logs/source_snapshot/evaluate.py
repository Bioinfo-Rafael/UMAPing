"""Evaluation only: ground truth never enters training."""
import numpy as np
import pandas as pd
from geometry import neighbors, shortcut_flags, STRUCTURES


def evaluate(frame,reference,query,yref,yquery):
    z=frame[['z0','z1']].to_numpy();ref=frame.iloc[reference].reset_index(drop=True)
    all_y=np.empty_like(z);all_y[reference]=yref;all_y[query]=yquery
    out=frame[['id','structure','branch','progress','split']].copy()
    rzi,rz=neighbors(z[reference],z[reference],15,np.arange(len(reference)))
    ryi,ry=neighbors(yref,yref,15,np.arange(len(reference)))
    scale=float(np.exp(np.median(np.log(rz[:,-1]/np.maximum(ry[:,-1],1e-12)))))
    for ids,selfidx in [(reference,np.arange(len(reference))),(query,None)]:
        src=frame.iloc[ids].reset_index(drop=True)
        iz,dz=neighbors(z[ids],z[reference],15,selfidx);iy,dy=neighbors(all_y[ids],yref,15,selfidx)
        for k in [5,15]:
            recall=np.array([len(set(a[:k])&set(b[:k]))/k for a,b in zip(iz,iy)])
            signed=np.log(np.maximum(scale*dy[:,k-1],1e-12)/dz[:,k-1])
            for name,value in [(f'recall{k}',recall),(f'rz{k}',dz[:,k-1]),(f'ry{k}',dy[:,k-1]),(f'log_radius{k}',signed),(f'abs_log_radius{k}',abs(signed))]:out.loc[ids,name]=value
        wrong_turn,wrong_branch=shortcut_flags(src,ref,iy,dz[:,-1]);out.loc[ids,'wrong_turn15']=wrong_turn.mean(1);out.loc[ids,'wrong_branch15']=wrong_branch.mean(1)
        bridge_edges=(src.structure.to_numpy()[:,None]>=3)|(ref.structure.to_numpy()[iz]>=3)
        hit=np.array([np.isin(a,b) for a,b in zip(iz,iy)])
        out.loc[ids,'bridge_true_edges15']=bridge_edges.sum(1)
        out.loc[ids,'bridge_lost_edges15']=(bridge_edges&~hit).sum(1)
        cross_bridge=bridge_edges&(src.structure.to_numpy()[:,None]!=ref.structure.to_numpy()[iz])
        out.loc[ids,'bridge_cross_true_edges15']=cross_bridge.sum(1)
        out.loc[ids,'bridge_cross_lost_edges15']=(cross_bridge&~hit).sum(1)
        out.loc[ids,'bridge_near']=(bridge_edges.any(1)).astype(int)
        out.loc[ids,'bridge_near_recall15']=np.where(bridge_edges.any(1),out.loc[ids,'recall15'],np.nan)
        for structure in sorted(src.structure.unique()):
            local=np.flatnonzero(src.structure.to_numpy()==structure);cand=np.flatnonzero(ref.structure.to_numpy()==structure)
            global_ids=ids[local]
            # Candidate set is exactly the same in z and y, restricted to structure.
            ex=None if selfidx is None else np.searchsorted(cand,selfidx[local])
            izw,dzw=neighbors(z[global_ids],z[reference[cand]],15,ex)
            iyw,dyw=neighbors(all_y[global_ids],yref[cand],15,ex)
            for k in [5,15]:
                recall=np.array([len(set(a[:k])&set(b[:k]))/k for a,b in zip(izw,iyw)])
                signed=np.log(np.maximum(scale*dyw[:,k-1],1e-12)/dzw[:,k-1])
                out.loc[global_ids,f'within_recall{k}']=recall
                out.loc[global_ids,f'within_log_radius{k}']=signed
                out.loc[global_ids,f'within_abs_log_radius{k}']=abs(signed)
    out['scale']=scale
    return out


def summarize(points):
    metrics=['recall5','recall15','within_recall5','within_recall15','log_radius5','log_radius15','abs_log_radius5','abs_log_radius15','within_log_radius15','within_abs_log_radius15','wrong_turn15','wrong_branch15','bridge_near_recall15']
    groups=[('all',np.ones(len(points),bool)),('internal',points.structure.to_numpy()<3)]
    groups += [(name,points.structure.to_numpy()==key) for key,name in STRUCTURES.items() if key in points.structure.unique()]
    groups += [(f'B_branch_{b}',(points.structure.to_numpy()==1)&(points.branch.to_numpy()==b)) for b in range(7)]
    groups += [('bridge_near',points.bridge_near.to_numpy()>0)]
    rows=[]
    for split in ['reference','query']:
        for group,mask in groups:
            p=points[mask&(points.split.to_numpy()==split)]
            if not len(p):continue
            d=dict(split=split,group=group,n=len(p),**{k:float(p[k].mean()) for k in metrics})
            for prefix in ['bridge','bridge_cross']:
                denom=p[prefix+'_true_edges15'].sum();d[prefix+'_edge_loss15']=float(p[prefix+'_lost_edges15'].sum()/denom) if denom else np.nan
                d[prefix+'_true_edges15']=int(denom)
            d['log_radius15_q10']=float(p.log_radius15.quantile(.1));d['log_radius15_q90']=float(p.log_radius15.quantile(.9));d['scale']=float(p.scale.iloc[0]);rows.append(d)
    return pd.DataFrame(rows)
