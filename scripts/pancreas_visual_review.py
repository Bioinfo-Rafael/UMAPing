#!/usr/bin/env python3
"""Plot pancreas results; saved-only by default. --regenerate explicitly opts
into frozen UMAPing inference, standard UMAP fit/transform, and fresh metrics.
No neural-network training, preprocessing refit, or downloads.

Embedding NPZ contract: reference (Nr,2), query (Nq,2), reference_index
(Nr,), query_index (Nq,). Indices explicitly refer to rows of the specified
PreparedDataset. Bare coordinate arrays without identities are rejected.
prepared_dataset_sha256 scalar, or the same key in <file>.json, binds the
row IDs to the exact dataset. Never invent IDs for an embedding.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, to_hex
from matplotlib.lines import Line2D
from PIL import Image, ImageOps, ImageDraw
import yaml

METHODS=('standard_umap','ours_full')
GENES=('INS','GCG','SST','KRT19')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def ids(values, name):
    values=np.asarray(values)
    require(values.ndim==1 and np.issubdtype(values.dtype,np.integer),f'{name}: explicit integer row IDs required')
    require(len(np.unique(values))==len(values) and np.all(values>=0),f'{name}: duplicate or negative IDs')
    return values.astype(np.int64)


def load_metrics(path):
    frame=pd.read_csv(path,dtype={'method':str,'label':str})
    require({'method','query_index','recall_at_5','recall_at_15','label'}<=set(frame),
            'advanced_per_query.csv requires method, query_index, label, Recall@5/15')
    result={}
    for method in METHODS:
        f=frame[frame.method==method].copy()
        require(len(f)>0,f'Missing method: {method}')
        f['query_index']=ids(f.query_index,f'{method} metrics query_index')
        require(f.label.notna().all(),f'{method}: missing celltype labels')
        require(np.isfinite(f[['recall_at_5','recall_at_15']]).all().all(),f'{method}: nonfinite Recall')
        require(((f[['recall_at_5','recall_at_15']]>=0)&(f[['recall_at_5','recall_at_15']]<=1)).all().all(),'Recall outside [0,1]')
        result[method]=f.set_index('query_index').sort_index()
    a,b=(result[m] for m in METHODS)
    require(a.index.equals(b.index),'Methods have different query ID sets; refusing silent intersection')
    require(a.label.equals(b.label),'Celltype labels differ for the same query IDs')
    return result


def load_prepared(path):
    with np.load(path,allow_pickle=False) as z:
        result={}
        for split in ('reference','query'):
            keys=[f'{split}_label__{col}' for col in ('celltype','tech')]
            require(set(keys)<=set(z.files),f'{split}: celltype/tech missing in PreparedDataset')
            labels={col:np.asarray(z[key]).astype(str) for col,key in zip(('celltype','tech'),keys)}
            require(all(v.ndim==1 for v in labels.values()),'Annotation arrays must be 1D')
            require(len(labels['celltype'])==len(labels['tech']),f'{split}: annotation lengths differ')
            require(z[f'{split}_features'].shape[0]==len(labels['tech']),f'{split}: feature/annotation lengths differ')
            result[split]=pd.DataFrame(labels,index=pd.Index(np.arange(len(labels['tech'])),name='index'))
            require(not result[split].isin(['','nan','None']).any().any(),f'{split}: missing annotations')
    return result


def load_embedding(path, prepared, metrics, prepared_hash):
    with np.load(path,allow_pickle=False) as z:
        required={'reference','query','reference_index','query_index'}
        require(required<=set(z.files),f'{path}: missing {sorted(required-set(z.files))}. Bare arrays or implicit reference row order are not accepted.')
        recorded=str(z['prepared_dataset_sha256'].item()) if 'prepared_dataset_sha256' in z else None
        sidecar=Path(str(path)+'.json')
        if sidecar.exists():
            external=json.loads(sidecar.read_text()).get('prepared_dataset_sha256')
            require(recorded is None or external is None or recorded==external,'Conflicting PreparedDataset hashes')
            recorded=recorded or external
        require(recorded is not None,f'{path}: PreparedDataset hash binding missing; explicit row IDs alone do not identify the source dataset')
        require(recorded==prepared_hash,f'{path}: PreparedDataset hash mismatch')
        frames={}
        for split in ('reference','query'):
            index=ids(z[f'{split}_index'],f'{path} {split}_index')
            y=np.asarray(z[split])
            require(y.shape==(len(index),2) and np.isfinite(y).all(),f'{path}: invalid {split} coordinates')
            require(np.array_equal(np.sort(index),prepared[split].index.to_numpy()),
                    f'{path}: {split} ID set must equal the full PreparedDataset split')
            coords=pd.DataFrame(y,columns=['x','y'],index=pd.Index(index,name='index'))
            frames[split]=prepared[split].join(coords,how='left',validate='one_to_one')
            for col in ('celltype','tech'):
                key=f'{split}_{col}'
                if key in z:
                    require(np.array_equal(np.asarray(z[key]).astype(str),prepared[split].loc[index,col].to_numpy()),
                            f'{path}: embedded {key} disagrees with indexed annotations')
        require(np.array_equal(metrics.index.to_numpy(),prepared['query'].index.to_numpy()),'Metric query IDs differ from PreparedDataset')
        require(np.array_equal(metrics.label.to_numpy(),prepared['query'].celltype.to_numpy()),'Metric celltype/PreparedDataset disagreement')
        frames['query']=frames['query'].join(metrics[['recall_at_5','recall_at_15']],validate='one_to_one')
        return frames,dict(explicit_reference_ids=True,explicit_query_ids=True,
                           prepared_hash_embedded=recorded is not None,alignment='explicit PreparedDataset row IDs; never embedding row order')


def palette(labels):
    labels=sorted(set(map(str,labels)))
    cmap=plt.get_cmap('tab20' if len(labels)<=20 else 'hsv')
    return {label:to_hex(cmap(i if len(labels)<=20 else i/max(len(labels),1))) for i,label in enumerate(labels)}


def bounds(frames, common_span=None):
    y=np.concatenate([f[['x','y']].to_numpy() for f in frames if len(f)])
    low=y.min(0); high=y.max(0); center=(low+high)/2
    span=max(float((high-low).max())*1.08,1e-5) if common_span is None else common_span
    return center,span


def set_view(ax, view):
    center,span=view
    ax.set_xlim(center[0]-span/2,center[0]+span/2)
    ax.set_ylim(center[1]-span/2,center[1]+span/2)
    ax.set_aspect('equal',adjustable='box')
    ax.tick_params(labelsize=7,color='#cccccc')
    ax.set_xlabel('Embedding dimension 1',fontsize=8); ax.set_ylabel('Embedding dimension 2',fontsize=8)
    for spine in ax.spines.values(): spine.set_color('#dddddd')


def points(ax,f,colors=None,**kwargs):
    if len(f): ax.scatter(f.x,f.y,c=colors,s=4,linewidths=0,rasterized=True,**kwargs)


class Review:
    def __init__(self,args):
        self.args=args; self.run=Path(args.run).resolve()
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        root=Path(args.repo_root).resolve() if args.repo_root else self.run.parents[2]
        self.output=Path(args.output).resolve() if args.output else root/'Final_analysis'/f'pancreas_batch_corrected_visual_review_{stamp}'
        require(not self.output.exists(),'Output already exists; refusing overwrite')
        require(self.output!=self.run and self.run not in self.output.parents,'Output may not be inside the source run')
        self.output.mkdir(parents=True); (self.output/'figures').mkdir(); (self.output/'tables').mkdir()
        self.inputs={}; self.catalog=[]; self.skipped=[]; self.checks=[]; self.data={}; self.annotations=None
        self.status='started'; self.teacher='unrecorded'; self.teacher_evidence=''; self.expression_info=''
        self.regeneration=None
        self.started=datetime.now(timezone.utc).isoformat(); self.names={'standard_umap':'Standard UMAP','ours_full':'UMAPing (teacher unrecorded)'}
        print(f'Output: {self.output}',flush=True)

    def track(self,path):
        path=Path(path).resolve(); self.inputs[str(path)]={'bytes':path.stat().st_size,'sha256':sha(path)}; return path

    def inventory(self):
        rows=[]
        for p in sorted(self.run.rglob('*')):
            if not p.is_file() or p.suffix not in ('.npz','.npy','.csv','.json','.yaml','.h5ad'): continue
            row=dict(path=str(p),bytes=p.stat().st_size,keys='')
            if p.suffix=='.npz':
                try:
                    with zipfile.ZipFile(p) as z: row['keys']=';'.join(n.removesuffix('.npy') for n in z.namelist())
                except Exception as e: row['keys']=f'unreadable: {e}'
            rows.append(row)
        pd.DataFrame(rows,columns=['path','bytes','keys']).to_csv(self.output/'tables/input_inventory.csv',index=False)

    def teacher_label(self):
        config=self.run/'config.yaml'
        if config.exists():
            cfg=yaml.safe_load(self.track(config).read_text())
            require(cfg.get('dataset',{}).get('name')=='pancreas','The input run is not pancreas')
            require(cfg['dataset'].get('params',{}).get('batch_correction') is True,'The input run is not batch-corrected pancreas')
        path=self.run/'metadata.json'; meta=json.loads(self.track(path).read_text()) if path.exists() else {}
        teacher=meta.get('repulsion_teacher')
        if self.args.teacher:
            require(not teacher or teacher==self.args.teacher,'Teacher option contradicts saved metadata')
            require(bool(self.args.teacher_evidence),'--teacher requires --teacher-evidence')
            teacher=self.args.teacher; evidence=self.args.teacher_evidence
        else: evidence='metadata.json:repulsion_teacher' if teacher else 'Run metadata does not record teacher; legacy pipeline default is Uniform MC, but not asserted as verified.'
        self.teacher=teacher or 'unrecorded'; self.teacher_evidence=evidence
        label={'uniform_mc':'Uniform MC teacher','fit_grid':'FitGrid teacher'}.get(self.teacher,'teacher unrecorded')
        self.names['ours_full']=f'UMAPing ({label})'

    def save(self,fig,stem,description,legends=()):
        if self.args.fixture: fig.suptitle('SYNTHETIC LAYOUT VALIDATION — NOT EXPERIMENTAL RESULTS',fontsize=10,color='darkred',y=.998)
        fig.canvas.draw(); renderer=fig.canvas.get_renderer(); box=fig.bbox
        for legend in legends:
            b=legend.get_window_extent(renderer)
            require(b.x0>=0 and b.y0>=0 and b.x1<=box.x1+1 and b.y1<=box.y1+1,f'Legend outside canvas: {stem}')
        for ext in ('png','pdf'): fig.savefig(self.output/'figures'/f'{stem}.{ext}',dpi=190,facecolor='white')
        plt.close(fig)
        self.catalog.append(dict(stem=stem,description=description,legend_bounds_checked=True))
        print(f'Saved: {stem} (PNG/PDF)',flush=True)

    def dotplot(self):
        counts=self.metrics['ours_full'].label.value_counts()
        pd.DataFrame({'celltype':counts.index,'n_query':counts.values,'included':counts.values>=100}).to_csv(
            self.output/'tables/zoom_eligibility.csv',index=False)
        rows=[]
        for celltype in sorted(self.metrics['ours_full'].label.unique()):
            for k in (5,15):
                a=self.metrics['standard_umap']; b=self.metrics['ours_full']; mask=a.label==celltype
                av=a.loc[mask,f'recall_at_{k}']; bv=b.loc[mask,f'recall_at_{k}']
                rows.append(dict(celltype=celltype,k=k,n_query=int(mask.sum()),standard_umap_mean=float(av.mean()),
                    umaping_mean=float(bv.mean()),paired_mean_difference=float((bv-av).mean())))
        table=pd.DataFrame(rows); table.to_csv(self.output/'tables/celltype_recall_comparison.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(12,max(5,.36*table.celltype.nunique()+1.8)),sharey=True)
        for ax,k in zip(axes,(5,15)):
            sub=table[table.k==k]; y=np.arange(len(sub))
            ax.hlines(y,sub.standard_umap_mean*100,sub.umaping_mean*100,color='#bbbbbb',lw=1)
            ax.scatter(sub.standard_umap_mean*100,y,facecolors='none',edgecolors='#256A9E',
                       linewidths=1.3,label=self.names['standard_umap'],s=40,zorder=3)
            ax.scatter(sub.umaping_mean*100,y,color='#CA692C',label=self.names['ours_full'],s=16,zorder=4)
            ax.set_yticks(y,[f'{r.celltype} (Q={r.n_query:,})' for r in sub.itertuples()])
            ax.set_xlabel(f'Mean Recall@{k} (%)'); ax.grid(axis='x',alpha=.18); ax.set_title(f'Recall@{k}: all cell types')
            ax.set_xlim(left=0)
        axes[0].invert_yaxis()
        handles,labels=axes[0].get_legend_handles_labels()
        leg=fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.57,.015),ncol=2,fontsize=9)
        fig.subplots_adjust(left=.25,right=.98,bottom=.16,top=.9,wspace=.18)
        self.save(fig,'05_celltype_recall_dotplot','全細胞型の平均Recall@5/15。IDで対応付けた同じquery。',(leg,))
        merged=self.metrics['standard_umap'][['label','recall_at_5','recall_at_15']].join(
            self.metrics['ours_full'][['recall_at_5','recall_at_15']],lsuffix='_standard_umap',rsuffix='_umaping',validate='one_to_one')
        for k in (5,15): merged[f'delta_recall_at_{k}']=merged[f'recall_at_{k}_umaping']-merged[f'recall_at_{k}_standard_umap']
        merged.to_csv(self.output/'tables/paired_query_recall.csv')
        self.paired=merged

    def embedding_path(self,method):
        explicit=self.args.standard_embedding if method=='standard_umap' else self.args.umaping_embedding
        if explicit: return Path(explicit)
        directory=Path(self.args.embedding_dir) if self.args.embedding_dir else self.run/'embeddings'
        aliases=('standard_umap','umap_transform') if method=='standard_umap' else ('ours_full','uniform_mc','umaping')
        found=[directory/f'{name}.npz' for name in aliases if (directory/f'{name}.npz').exists()]
        require(len(found)<=1,f'Ambiguous {method} embedding candidates; use explicit path option')
        return found[0] if found else directory/f'{aliases[0]}.npz'

    def load_coordinates(self):
        prepared=Path(self.args.prepared) if self.args.prepared else self.run/'cache/prepared_dataset.npz'
        needed=[prepared]+[self.embedding_path(m) for m in METHODS]
        missing=[str(p) for p in needed if not p.is_file()]
        if prepared.is_file():
            self.annotations=load_prepared(self.track(prepared))
            query=self.annotations['query']; reference=self.annotations['reference']
            for method,frame in self.metrics.items():
                require(np.array_equal(frame.index,query.index),f'{method}: metric IDs differ from PreparedDataset')
                require(np.array_equal(frame.label,query.celltype),f'{method}: metric celltype differs from PreparedDataset')
            self.checks.append(dict(scope='annotations/metrics',explicit_query_ids=True,celltype_verified=True))
            for split,frame in self.annotations.items():
                frame.to_csv(self.output/'tables'/f'{split}_annotations.csv')
            config_path=self.run/'config.yaml'
            if config_path.exists():
                cfg=yaml.safe_load(self.track(config_path).read_text()); expected=cfg.get('dataset',{}).get('params',{}).get('query_tech')
                if expected is not None:
                    require(set(query.tech)==set(expected),'Actual query tech differs from saved query_tech')
                    require(not set(reference.tech)&set(expected),'Reference contains a held-out query technology')
        if missing:
            self.skipped.append(dict(item='01/02/03/04/06 coordinate panels',reason='Missing saved inputs',paths=missing))
            return False
        digest=self.inputs[str(prepared.resolve())]['sha256']
        query=self.annotations['query']; reference=self.annotations['reference']
        for method in METHODS:
            path=self.track(self.embedding_path(method)); sidecar=Path(str(path)+'.json')
            if sidecar.exists(): self.track(sidecar)
            self.data[method],check=load_embedding(path,self.annotations,self.metrics[method],digest)
            self.checks.append(dict(method=method,**check))
            for split,frame in self.data[method].items():
                frame.to_csv(self.output/'tables'/f'aligned_{split}_{method}.csv',index_label=f'{split}_index')
        self.views={m:bounds(list(self.data[m].values())) for m in METHODS}
        self.palettes={col:palette(pd.concat([reference[col],query[col]])) for col in ('celltype','tech')}
        for col,mapping in self.palettes.items():
            pd.DataFrame([dict(label=k,color=v,n_reference=int((reference[col]==k).sum()),n_query=int((query[col]==k).sum()))
                          for k,v in mapping.items()]).to_csv(self.output/'tables'/f'{col}_colors_counts.csv',index=False)
        return True

    def overview(self,column,number):
        fig,axes=plt.subplots(3,2,figsize=(13,14)); colors=self.palettes[column]
        for j,method in enumerate(METHODS):
            ref=self.data[method]['reference']; query=self.data[method]['query']
            for i,role in enumerate(('Reference only','Query only','Reference (gray) + query')):
                ax=axes[i,j]
                if i==2: points(ax,ref,'#bfbfbf',alpha=.23)
                active=ref if i==0 else query
                points(ax,active,active[column].map(colors).tolist(),alpha=.8)
                set_view(ax,self.views[method])
                ax.set_title(f'{self.names[method]}\n{role} | R={len(ref) if i!=1 else 0:,}; Q={len(query) if i!=0 else 0:,}',fontsize=10)
        handles=[Line2D([],[],marker='o',linestyle='',markersize=4,color=c,label=label) for label,c in colors.items()]
        legend=fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.79,.5),title=column,fontsize=9)
        fig.subplots_adjust(left=.06,right=.78,bottom=.055,top=.95,hspace=.30,wspace=.20)
        self.save(fig,f'{number:02d}_overview_by_{column}',f'3行×2列、{column}色。同じ手法の座標範囲は全行固定。',(legend,))

    def zooms(self):
        counts=self.annotations['query'].celltype.value_counts()
        eligible=sorted(counts[counts>=100].index)
        pd.DataFrame({'celltype':counts.index,'n_query':counts.values,'included':counts.values>=100}).to_csv(self.output/'tables/zoom_eligibility.csv',index=False)
        for order,celltype in enumerate(eligible,1):
            subsets={m:{s:f[f.celltype==celltype] for s,f in self.data[m].items()} for m in METHODS}
            local={m:bounds(list(subsets[m].values())) for m in METHODS}
            span=max(v[1] for v in local.values())
            fig,axes=plt.subplots(1,2,figsize=(12,6)); colors=self.palettes['tech']
            for ax,method in zip(axes,METHODS):
                ref=subsets[method]['reference']; query=subsets[method]['query']
                points(ax,ref,'#bcbcbc',alpha=.4); points(ax,query,query.tech.map(colors).tolist(),alpha=.85)
                set_view(ax,(local[method][0],span))
                ax.set_title(f'{self.names[method]}\n{celltype} | R={len(ref):,}, Q={len(query):,}\nMean Recall@15={query.recall_at_15.mean():.2%}',fontsize=10)
            handles=[Line2D([],[],marker='o',linestyle='',color='#bcbcbc',label='Reference (target cell type)',markersize=4)]
            handles += [Line2D([],[],marker='o',linestyle='',color=c,label=k,markersize=4) for k,c in colors.items()]
            legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.075),ncol=min(3,len(handles)),fontsize=8)
            fig.text(.5,.015,f'Equal numeric axis span in both panels: {span:.3g}. Native coordinates; visual compactness is not a density metric.',ha='center',fontsize=8)
            fig.subplots_adjust(left=.075,right=.975,bottom=.25,top=.8,wspace=.22)
            slug=''.join(c if c.isalnum() else '_' for c in celltype)
            self.save(fig,f'03_{order:02d}_celltype_zoom_{slug}',f'{celltype}: query>=100の事前規則で対象。両手法の表示幅を統一。',(legend,))

    def recall_maps(self,difference=False):
        fig,axes=plt.subplots(2,2,figsize=(12,11)); norm=Normalize(0,1); cmap='viridis'
        if difference:
            limit=max(float(self.paired[[f'delta_recall_at_{k}' for k in (5,15)]].abs().max().max()),1/15)
            norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit); cmap='RdBu_r'
        for i,k in enumerate((5,15)):
            for j,method in enumerate(METHODS):
                ax=axes[i,j]; ref=self.data[method]['reference']; query=self.data[method]['query']
                points(ax,ref,'#bfbfbf',alpha=.22)
                values=self.paired.loc[query.index,f'delta_recall_at_{k}'] if difference else query[f'recall_at_{k}']
                points(ax,query,values.to_numpy(),cmap=cmap,norm=norm,alpha=1)
                set_view(ax,self.views[method])
                ax.set_title(f'{self.names[method]}\n{"Paired delta " if difference else ""}Recall@{k} | R={len(ref):,}, Q={len(query):,}',fontsize=10)
        fig.subplots_adjust(left=.07,right=.85,bottom=.065,top=.93,hspace=.28,wspace=.18)
        cax=fig.add_axes([.89,.18,.018,.60]); bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax)
        bar.set_label('UMAPing minus Standard UMAP Recall (same query)' if difference else 'Recall (0 to 1)')
        self.save(fig,'06_paired_recall_difference_on_both_embeddings' if difference else '04_query_recall_at_5_and_15',
                  '同一queryの差を両座標へ表示。共通発散色域。' if difference else '保存済みRecall@5/15を共通0–1色域で表示。')

    def expression(self):
        if not self.args.expression_csv:
            self.expression_info='省略: 対応IDと発現処理を確認できるマーカー発現表が指定されていない。PCA/scVI特徴量やcelltypeから発現量は復元しない。'
            self.skipped.append(dict(item='07 marker expression',reason=self.expression_info)); return
        require(self.args.expression_kind in ('counts','log1p_cptt'),'--expression-kind counts or log1p_cptt is required')
        require(bool(self.args.expression_description),'--expression-description is required')
        sidecar=Path(str(self.args.expression_csv)+'.json')
        require(sidecar.is_file(),'Marker expression requires <csv>.json with prepared_dataset_sha256')
        expression_meta=json.loads(self.track(sidecar).read_text())
        prepared=Path(self.args.prepared) if self.args.prepared else self.run/'cache/prepared_dataset.npz'
        require(expression_meta.get('prepared_dataset_sha256')==self.inputs[str(prepared.resolve())]['sha256'],
                'Expression table is not bound to the same PreparedDataset')
        f=pd.read_csv(self.track(self.args.expression_csv),dtype={'split':str})
        require({'split','index',*GENES}<=set(f),'Marker CSV requires split,index,INS,GCG,SST,KRT19')
        require(set(f.split)=={'reference','query'} and not f.duplicated(['split','index']).any(),'Invalid/duplicate expression split IDs')
        require(np.isfinite(f[list(GENES)]).all().all() and (f[list(GENES)]>=0).all().all(),'Invalid expression values')
        for split in ('reference','query'):
            index=ids(f.loc[f.split==split,'index'],f'{split} expression IDs')
            require(np.array_equal(np.sort(index),self.annotations[split].index),'Expression IDs do not match annotation IDs')
        if self.args.expression_kind=='counts':
            require('total_counts' in f,'counts input requires full-transcriptome total_counts, not sum of four markers')
            require(np.isfinite(f.total_counts).all() and (f.total_counts>0).all(),'Invalid full-transcriptome library sizes')
            require((f[list(GENES)].sum(axis=1)<=f.total_counts+1e-6).all(),'Marker counts exceed library size')
            for gene in GENES:
                f[gene]=np.log1p(f[gene].to_numpy(dtype=float)/f.total_counts.to_numpy()*1e4)
        self.expression_info=f'log1p(counts / full-transcriptome library size × 10,000); source kind={self.args.expression_kind}; {self.args.expression_description}. 全細胞共通の遺伝子別0–最大値色域。'
        f.to_csv(self.output/'tables/aligned_marker_expression_log1p_cptt.csv',index=False)
        for order,gene in enumerate(GENES,1):
            norm=Normalize(0,max(float(f[gene].max()),1e-12)); fig,axes=plt.subplots(2,2,figsize=(12,11))
            for i,split in enumerate(('reference','query')):
                expression=f[f.split==split].set_index('index')[gene]
                for j,method in enumerate(METHODS):
                    cells=self.data[method][split]; values=expression.loc[cells.index]
                    points(axes[i,j],cells,values.to_numpy(),cmap='viridis',norm=norm,alpha=1)
                    set_view(axes[i,j],self.views[method])
                    axes[i,j].set_title(f'{self.names[method]}\n{gene} | {split}, n={len(cells):,}',fontsize=10)
            fig.subplots_adjust(left=.07,right=.85,bottom=.06,top=.93,hspace=.28,wspace=.18)
            cax=fig.add_axes([.89,.18,.018,.60]); fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),cax=cax,label=f'{gene}: log1p counts per 10,000')
            self.save(fig,f'07_{order:02d}_marker_expression_{gene}',f'{gene}: reference/query別。同じ正規化と共通色域。')

    def contact_sheets(self):
        # Explicit QA aids, not a claim that a person/agent inspected the images.
        paths=sorted((self.output/'figures').glob('*.png'))
        for start in range(0,len(paths),4):
            sheet=Image.new('RGB',(1200,1260),'white'); draw=ImageDraw.Draw(sheet)
            for j,p in enumerate(paths[start:start+4]):
                with Image.open(p) as img: thumb=ImageOps.contain(img.convert('RGB'),(590,580))
                x=(j%2)*600; y=(j//2)*630; sheet.paste(thumb,(x+(600-thumb.width)//2,y+35))
                draw.text((x+8,y+8),p.stem[:70],fill='black')
            sheet.save(self.output/'tables'/f'qa_contact_sheet_{start//4+1:02d}.png')

    def finish(self,error=None):
        for path,meta in self.inputs.items(): require(sha(path)==meta['sha256'],f'Input changed while plotting: {path}')
        meta=dict(status=self.status,started_utc=self.started,ended_utc=datetime.now(timezone.utc).isoformat(),
                  script_sha256=sha(Path(__file__)),versions={'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
                  run=str(self.run),teacher=self.teacher,teacher_evidence=self.teacher_evidence,inputs=self.inputs,
                  identity_checks=self.checks,skipped=self.skipped,error=error,figures=self.catalog,
                  no_training=self.regeneration is None,no_inference=self.regeneration is None,
                  no_metric_recomputation=self.regeneration is None,neural_network_training=False,
                  regeneration=self.regeneration,
                  visual_review='Legend bounds checked; contact sheets provided. Actual image inspection must be recorded separately.',
                  command=['python',str(Path(__file__).resolve()),*sys.argv[1:]])
        json_write(self.output/'tables/provenance.json',meta)
        pd.DataFrame(self.catalog).to_csv(self.output/'tables/figure_catalog.csv',index=False)
        self.contact_sheets()
        observed='座標パネルは未作成のため、配置・重なりの違いは観察できません。'
        if self.data:
            observed='reference/query、celltype、tech、Recallおよび同一queryの差を分けて表示しました。配置・重なりは図から確認できますが、手法間の位置・回転自体を性能差とは解釈しません。'
        numeric=[]
        if hasattr(self,'paired'):
            for k in (5,15):
                a=self.paired[f'recall_at_{k}_standard_umap'].mean(); b=self.paired[f'recall_at_{k}_umaping'].mean()
                numeric.append(f'- Recall@{k}: 通常UMAP {a:.3%}、UMAPing {b:.3%}、paired平均差 {(b-a)*100:+.3f} percentage points（同じquery {len(self.paired):,}点）。')
        tech='未検証（PreparedDatasetの実ラベルをまだ読み取れていません）。config上の予定値を実測値とは扱いません。'
        if self.annotations:
            tech=', '.join(f'{k}: {v:,} query' for k,v in self.annotations['query'].tech.value_counts().sort_index().items())
        lines=['# Pancreas batch-corrected visual review','',f'状態: **{self.status}**。',f'入力run: `{self.run}`。',
               f'UMAPing teacher: **{self.teacher}**。根拠: {self.teacher_evidence}',
               f'queryの測定技術: {tech}','',
               '## 対応関係の検証',
               'per-query metricsはmethod別のquery_indexで一対一結合し、ID集合・重複・celltypeの一致を検証。座標は明示的なreference_index/query_indexでPreparedDatasetのラベルへ結合。長さが同じという理由だけでは結合しません。',
               'indexはそのPreparedDataset内の行IDです。元の生物学的cell barcodeとは区別します。座標ファイルまたはsidecarに保存されたPreparedDatasetのSHA256も必須として照合します。IDの無い旧形式座標からIDを生成しません。',
               'teacherが未記録なら未確認と表記します。旧pipelineの既定はUniform MCですが、この情報だけで既存runをFitGridまたは検証済みUniformとは表記しません。','',
               '## 図一覧']
        if self.regeneration is not None:
            lines[3:3]=['**座標の再生成モードです。** 通常UMAPは保存済みreference特徴でfitしてqueryをtransformし、UMAPingは既存の凍結チェックポイントからqueryのみを再推論する構成です。reference軌道は変更しません。完了段階と成否は `tables/regenerated/regeneration.json` を参照してください。',
                        '前処理・scVI/scArches・retriever・Spectral・反発ネットの再学習はありません。Recallは今回の座標に対して再評価し、以前のmetricsを今回の座標図に流用していません。',
                        'IDは再計算前にPreparedDatasetの各行へ割り当て、推論ループまたはfit/transformの入力出力対応から保存しました。旧座標の行順を推測して付与したIDではありません。','']
        for fig in self.catalog:
            s=fig['stem']; lines.append(f'- [{s} PNG](figures/{s}.png) / [PDF](figures/{s}.pdf): {fig["description"]}')
        lines+=['','## 観察できる差',observed,'','## 数値で裏付けられる差',*numeric,
                ('平均値は今回再生成した座標の再評価です。旧実行との数値的一致は保証せず、ライブラリ版とseedを記録します。' if self.regeneration is not None else '平均値は保存済み評価の記述統計です。')+'新しい統計検定や密度指標は追加していません。','',
                '## まだ言えないこと',
                'クラスターの位置、回転、コンパクトさだけから生物学的優劣・密度改善は判断できません。拡大図はquery>=100の全細胞型を選び、良い結果の細胞型だけを選択しません。表示幅を左右で一致させても、手法間の座標スケールが生物学的に等しいことを保証しません。',
                'batch-corrected特徴空間での近傍保持と、生物学的なbatch除去・celltype保存は別の評価です。celltypeやtechの混ざり方だけから因果的な改善は主張しません。',
                '教師の比較実験ではないためUniform対FitGridの効果は判断できません。','',
                '## マーカー発現',self.expression_info or '座標入力が不足しており未実行。','',
                '## 不足・省略']
        for item in self.skipped: lines.append('- '+json.dumps(item,ensure_ascii=False))
        if error: lines.append('- ERROR: '+error)
        lines+=['','## 入力ファイル',*[f'- `{p}` (SHA256 `{v["sha256"]}`)' for p,v in self.inputs.items()],
                '','全候補一覧: [input_inventory.csv](tables/input_inventory.csv)。','',
                '## 再実行コマンド','```bash',shlex.join(meta['command']),'```',
                '同名出力は拒否します。再実行時は--outputを省略するか、新規ディレクトリを指定してください。','',
                '## 画像確認',
                '図外凡例のcanvas内収まりを描画後に検査。PNG/PDFを出力し、tables/qa_contact_sheet_*.pngを作成。実画像を確認した記録は別途保存してください。自動検査を目視確認とは呼びません。','']
        (self.output/'README.md').write_text('\n'.join(lines))

    def execute(self):
        self.inventory(); self.teacher_label()
        if self.args.regenerate:
            require(not any((self.args.metrics,self.args.prepared,self.args.embedding_dir,self.args.standard_embedding,self.args.umaping_embedding)),
                    '--regenerate cannot mix explicit external metrics/embedding/prepared overrides')
            from pancreas_regenerate import regenerate
            self.regeneration={'status':'started','path':str(self.output/'tables/regenerated')}
            generated=regenerate(self.run,self.output/'tables/regenerated',device=self.args.device,
                                  threads=self.args.threads,max_seconds=self.args.max_seconds)
            self.regeneration=json.loads(self.track(generated/'regeneration.json').read_text())
            self.args.metrics=str(generated/'advanced_per_query.csv')
            self.args.embedding_dir=str(generated/'embeddings')
        metrics_path=Path(self.args.metrics) if self.args.metrics else self.run/'metrics/advanced_per_query.csv'
        self.metrics=load_metrics(self.track(metrics_path))
        self.checks.append(dict(scope='paired metrics',query_ids_equal=True,unique_ids=True,celltype_equal=True,n_queries=len(self.metrics['ours_full'])))
        have_coordinates=False
        try: have_coordinates=self.load_coordinates()
        except (ValueError,KeyError) as e:
            self.skipped.append(dict(item='coordinate panels',reason=str(e)))
            self.data={}
        if have_coordinates:
            self.overview('celltype',1); self.overview('tech',2); self.zooms(); self.recall_maps()
        self.dotplot()
        if have_coordinates:
            self.recall_maps(difference=True)
            try: self.expression()
            except (ValueError,KeyError) as e:
                self.expression_info='省略: '+str(e); self.skipped.append(dict(item='07 marker expression',reason=str(e)))
        self.status=('complete_regenerated_required_plots' if self.regeneration is not None else 'complete_required_plots') if have_coordinates else 'partial_missing_or_unverified_coordinates'
        self.finish()
        print(f'Status: {self.status}\nREADME: {self.output / "README.md"}',flush=True)
        return 0 if have_coordinates else 2


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True); p.add_argument('--repo-root'); p.add_argument('--output')
    p.add_argument('--prepared'); p.add_argument('--embedding-dir')
    p.add_argument('--metrics',help='Explicit matching per-query metrics; never use old metrics for regenerated coordinates')
    p.add_argument('--regenerate',action='store_true',help='Authorize standard UMAP refit, frozen UMAPing inference and fresh metrics')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--threads',type=int,default=1)
    p.add_argument('--max-seconds',type=float,default=7200,help='Regeneration time limit; default 2 hours')
    p.add_argument('--standard-embedding'); p.add_argument('--umaping-embedding')
    p.add_argument('--teacher',choices=['uniform_mc','fit_grid']); p.add_argument('--teacher-evidence')
    p.add_argument('--expression-csv'); p.add_argument('--expression-kind',choices=['counts','log1p_cptt'])
    p.add_argument('--expression-description'); p.add_argument('--fixture',action='store_true',help=argparse.SUPPRESS)
    return p


def main():
    args=parser().parse_args(); app=Review(args)
    try: return app.execute()
    except Exception as e:
        regeneration_manifest=app.output/'tables/regenerated/regeneration.json'
        if args.regenerate and regeneration_manifest.exists():
            app.regeneration=json.loads(app.track(regeneration_manifest).read_text())
        app.status='failed'; app.finish(error=str(e)); traceback.print_exc(); return 1

if __name__=='__main__': raise SystemExit(main())
