"""Rebuild publication figures from saved embeddings and point metrics only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm
from matplotlib.lines import Line2D
from geometry import STRUCTURES,neighbors
plt.rcParams.update({'font.size':9,'axes.titlesize':11,'figure.dpi':120,'pdf.fonttype':42,'savefig.bbox':'tight'})
METHODS=['truth','pca','umap','uniform','fitgrid']
NAMES=['Ground truth 2D','PCA (reference fit)','UMAP fit / transform','UMAPing Uniform','UMAPing FitGrid']
COLORS=['#2166ac','#d6604d','#4d9221','#984ea3','#dfaa00']
BRANCH_COLORS=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666','#1f78b4','#b2df8a','#fb9a99','#cab2d6','#fdbf6f']

SAVE_PDF = True

def save(fig,base):
    base=Path(base);base.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(base.with_suffix('.png'),dpi=165)
    if SAVE_PDF:fig.savefig(base.with_suffix('.pdf'))
    plt.close(fig)

def truth_figures(run):
    full=pd.read_csv(run/'data/sparse_bridge/points.csv')
    present=set(full.structure);internal_count=int((full.structure<3).sum());bridge_count=int((full.structure>=3).sum())
    zfull=full[['z0','z1']].to_numpy();lo,hi=bounds(zfull);xlim=(lo[0],hi[0]);ylim=(lo[1],hi[1])
    fig,axes=plt.subplots(2,3,figsize=(16,6),layout='constrained')
    for row,condition in enumerate(['disconnected','sparse_bridge']):
        df=pd.read_csv(run/'data'/condition/'points.csv');z=df[['z0','z1']].to_numpy();ref=np.flatnonzero(df.split=='reference')
        _,d=neighbors(z,z[ref],15);_,dr=neighbors(z[ref],z[ref],15,np.arange(len(ref)));d[ref]=dr;density=np.log10(15/(np.pi*np.maximum(d[:,-1],1e-12)**2))
        for col in range(3):
            ax=axes[row,col]
            if col==0:
                for s,name in STRUCTURES.items():
                    mask=df.structure==s
                    if mask.any():ax.scatter(*z[mask].T,s=3,c=COLORS[s],label=name,rasterized=True)
            else:
                c=df.progress if col==1 else density;im=ax.scatter(*z.T,s=3,c=c,cmap='viridis',vmin=0 if col==1 else -1,vmax=1 if col==1 else 3.5,rasterized=True)
                fig.colorbar(im,ax=ax,shrink=.65,label=('Within-structure progress / ellipse x' if 2 in present else 'Arclength / within-branch progress') if col==1 else 'log10(15 / pi r15^2), diagnostic')
            ax.set_aspect('equal');ax.set_xlim(xlim);ax.set_ylim(ylim);ax.set_title(condition+' | '+['structure','progress','local density'][col]);ax.set_xlabel('z0');ax.set_ylabel('z1')
    handles=[Line2D([],[],marker='o',ls='',color=COLORS[s],label=name) for s,name in STRUCTURES.items() if s in present]
    fig.legend(handles=handles,loc='outside lower center',ncol=5,fontsize=9,frameon=False)
    fig.suptitle(f'Fixed geometry BEFORE training | {internal_count:,} shared internal points + {bridge_count} bridge points',fontsize=14)
    save(fig,run/'figures/01_ground_truth_geometry')
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    from matplotlib.collections import LineCollection
    audit=json.loads((run/'metrics/input_graph_audit.json').read_text())
    for ax,condition in zip(axes,['disconnected','sparse_bridge']):
        df=pd.read_csv(run/'data'/condition/'points.csv');a=np.load(run/'data'/condition/'arrays.npz');ref=a['reference'];z=a['z'][ref];knn=np.load(run/'data'/condition/'reference_knn.npz')['indices']
        seg=np.stack([np.repeat(z,15,axis=0),z[knn.ravel()]],axis=1)
        ax.add_collection(LineCollection(seg,colors='#777777',linewidths=.15,alpha=.3,rasterized=True));ax.scatter(*z.T,c=[COLORS[s] for s in df.structure.to_numpy()[ref]],s=3,rasterized=True)
        item=audit[condition+'_reference'];ax.set_title(f'{condition}: {item["components"]} components, no flagged shortcuts');ax.set(xlim=xlim,ylim=ylim,xlabel='z0',ylabel='z1');ax.set_aspect('equal')
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=COLORS[s],label=name) for s,name in STRUCTURES.items() if s in present],loc='outside lower center',ncol=5,frameon=False)
    fig.suptitle('Input exact kNN graph | reference only, k=15 excluding self',fontsize=14);save(fig,run/'figures/02_input_reference_graph')


def bounds(y):
    lo=np.min(y,axis=0);hi=np.max(y,axis=0);span=np.maximum(hi-lo,.2);return lo-.06*span,hi+.06*span

def panels(run,condition,seed,kind,number,subset=None,suffix=''):
    data=run/'data'/condition;df=pd.read_csv(data/'points.csv');a=np.load(data/'arrays.npz');ref,q=a['reference'],a['query'];coords={'truth':a['z']};points={}
    for m in METHODS[1:]:
        v=np.load(run/'embeddings'/condition/f'seed_{seed}'/(m+'.npz'));yy=np.empty((len(df),2));yy[ref]=v['reference'];yy[q]=v['query'];coords[m]=yy
        points[m]=pd.read_csv(run/'metrics'/condition/f'seed_{seed}'/(m+'_points.csv.gz'))
    true=points['pca'];selected=np.ones(len(df),bool) if subset is None else subset(df)
    allbounds={m:bounds(y[selected]) for m,y in coords.items()}
    joint=bounds(np.concatenate([coords['uniform'][selected],coords['fitgrid'][selected]]));allbounds['uniform']=allbounds['fitgrid']=joint
    fig,axes=plt.subplots(2,5,figsize=(20,8.5),layout='constrained');legend=[];norm=None;cmap=None
    if kind=='structure':
        colors=np.array([COLORS[s] for s in df.structure]);legend=[Line2D([],[],marker='o',ls='',color=COLORS[s],label=name,markersize=5) for s,name in STRUCTURES.items() if (df.structure==s).any()]
    elif kind=='branch':
        labels=np.array(['A turn '+str(b) if s==0 else 'B branch '+str(b) if s==1 else STRUCTURES[s] for s,b in zip(df.structure,df.branch)])
        unique=sorted(set(labels));mapping={v:BRANCH_COLORS[i] for i,v in enumerate(unique)};colors=np.array([mapping[l] for l in labels]);legend=[Line2D([],[],marker='o',ls='',color=mapping[l],label=l,markersize=5) for l in unique]
    elif kind=='progress':norm=Normalize(0,1);cmap='viridis';clabel='A: arclength; B: within branch; '+('C: normalized x; ' if (df.structure==2).any() else '')+'bridges: arclength'
    elif kind=='density':
        density=np.log10(15/(np.pi*true.rz15.to_numpy()**2));norm=Normalize(-1,3.5);cmap='viridis';clabel='Ground-truth log10(15 / pi r15^2); same colors across methods'
    elif kind=='error':norm=TwoSlopeNorm(vmin=-2,vcenter=0,vmax=2);cmap='coolwarm';clabel='ln(s r_y15 / r_z15): negative = compression, positive = expansion (clipped colors)'
    elif kind=='recall':norm=Normalize(0,1);cmap='viridis';clabel='Recall@15 to reference; reference row excludes self, query row is out-of-sample'
    for col,(m,title) in enumerate(zip(METHODS,NAMES)):
        if kind in ['structure','branch']:c=colors
        elif kind=='progress':c=df.progress.to_numpy()
        elif kind=='density':c=density
        elif kind=='error':c=np.zeros(len(df)) if m=='truth' else points[m].log_radius15.to_numpy()
        else:c=np.ones(len(df)) if m=='truth' else points[m].recall15.to_numpy()
        for row,ids in enumerate([ref,q]):
            ids=ids[selected[ids]];ax=axes[row,col];y=coords[m];kwargs={} if norm is None else dict(norm=norm,cmap=cmap)
            im=ax.scatter(*y[ids].T,c=c[ids],s=4 if subset is None else 8,linewidths=0,rasterized=True,**kwargs)
            lo,hi=allbounds[m];ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_title(title if row==0 else f'query n={len(ids):,}');ax.tick_params(labelsize=7)
            if col==0:ax.set_ylabel('Reference only' if row==0 else 'Query only',fontsize=12)
    fig.suptitle(f'{condition} | matched learning seed {seed} | {kind}'+(f' | {suffix}' if suffix else ''),fontsize=15)
    if legend:fig.legend(handles=legend,loc='outside lower center',ncol=min(7,len(legend)),fontsize=9,frameon=False)
    else:fig.colorbar(im,ax=axes.ravel().tolist(),location='bottom',fraction=.04,pad=.05,aspect=75,label=clabel,extend='both' if kind in ['error','density'] else 'neither')
    save(fig,run/'figures'/condition/f'seed_{seed}'/f'{number:02d}_{kind}{suffix}')


def all_figures(run):
    files=sorted((run/'metrics').glob('*/seed_*/summary.csv'))
    if not files:return
    summary=pd.concat([pd.read_csv(p) for p in files],ignore_index=True);summary.to_csv(run/'metrics/all_summary.csv',index=False)
    for file in files:
        condition=file.parents[1].name;seed=int(file.parent.name.split('_')[1]);out=run/'figures'/condition/f'seed_{seed}'
        for n,kind in enumerate(['structure','branch','progress','density','error','recall'],10):
            if not (out/f'{n:02d}_{kind}.png').exists():panels(run,condition,seed,kind,n)
        for n,label,select in [(20,'_spiral_zoom',lambda d:d.structure==0),(21,'_branch_zoom',lambda d:(d.structure==1)&(d.z0<.4)&(d.z1>-.8)),(22,'_bridge_zoom',lambda d:(d.structure>=3)|((d.z1>2)&(d.structure==1))|((d.z0>-9)&(d.structure==0))|((d.z0<11.7)&(d.structure==2)))]:
            if n==22 and condition=='disconnected':continue
            if not (out/f'{n:02d}_progress{label}.png').exists():panels(run,condition,seed,'progress',n,subset=select,suffix=label)
    # Points show every seed; bars are seed means, never best-seed selection.
    groups=['A_spiral','B_tree','C_ellipse','bridge_AB','bridge_BC'];methods=['pca','umap','uniform','fitgrid'];mcolors=['#555555','#2166ac','#e08214','#762a83']
    for cond in ['disconnected','sparse_bridge']:
        sub=summary[(summary.condition==cond)&summary.group.isin(groups)];gg=[g for g in groups if g in sub.group.unique()]
        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        for row,split in enumerate(['reference','query']):
            for col,(metric,title) in enumerate([('within_recall15','Within-structure Recall@15 (higher better)'),('abs_log_radius15','Mean absolute log radius error (lower better)'),('log_radius15','Signed log radius error (0 = agreement)')]):
                ax=axes[row,col]
                for mi,m in enumerate(methods):
                    ss=sub[(sub.method==m)&(sub.split==split)];means=ss.groupby('group')[metric].mean().reindex(gg);xs=np.arange(len(gg))+(mi-1.5)*.18
                    ax.bar(xs,means,width=.17,color=mcolors[mi],alpha=.65,label=NAMES[mi+1])
                    for j,g in enumerate(gg):
                        vals=ss.loc[ss.group==g,metric].to_numpy();ax.scatter(np.full(len(vals),xs[j]),vals,s=15,color=mcolors[mi],edgecolor='white',linewidth=.4,zorder=3)
                ax.axhline(0,color='black',lw=.6);ax.set_xticks(np.arange(len(gg)),gg,rotation=25,ha='right');ax.set_title(split+' | '+title)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=4,frameon=False);fig.suptitle(cond+' | structure metrics, all executed seeds',fontsize=15);save(fig,run/'figures'/f'30_{cond}_structure_metrics')
    # Connectivity and bridge losses must accompany the apparent island separation.
    sub=summary[(summary.condition=='sparse_bridge')&(summary.group=='all')]
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for ax,metric,title in zip(axes,['bridge_edge_loss15','bridge_cross_edge_loss15','wrong_turn15'],['Missing true bridge-related edges','Missing true bridge endpoint edges','False spiral shortcuts / 15 neighbors']):
        for mi,m in enumerate(methods):
            for si,split in enumerate(['reference','query']):
                values=sub.loc[(sub.method==m)&(sub.split==split),metric].to_numpy();pos=mi+(si-.5)*.26
                ax.bar(pos,np.mean(values),width=.25,color=mcolors[mi],alpha=.35 if si==0 else .85,hatch='//' if si==0 else None)
                ax.scatter(np.full(len(values),pos),values,color=mcolors[mi],s=15,zorder=4)
        ax.set_xticks(range(4),['PCA','UMAP','Uniform','FitGrid']);ax.set_title(title);ax.set_ylabel('Fraction (lower better)');ax.set_ylim(bottom=0)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor='gray',alpha=.35,hatch='//',label='Reference'),Patch(facecolor='gray',alpha=.85,label='Query')],loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('sparse_bridge | connectivity, all executed seeds',fontsize=14);save(fig,run/'figures/31_bridge_connectivity_metrics')
