import numpy as np
import pandas as pd
from geometry import generate,neighbors,shortcut_flags,EDGES,NODES
from evaluate import evaluate

def test_geometry_identity_and_isometry():
    f,x,A=generate();internal=f[f.structure<3]
    assert len(internal)==6000 and (internal.split=='reference').sum()==4000
    assert (f.split=='reference').sum()==4320
    np.testing.assert_allclose(A.T@A,np.eye(2),atol=1e-14)
    z=f[['z0','z1']].to_numpy();np.testing.assert_allclose(np.linalg.norm(x[::10]-x[1::10],axis=1),np.linalg.norm(z[::10]-z[1::10],axis=1),atol=1e-13)

def test_metric_scale_rotation_invariance_and_self_exclusion():
    f,_,_=generate();f=f.iloc[:150].copy().reset_index(drop=True);ref=np.flatnonzero(f.split=='reference');q=np.flatnonzero(f.split=='query');z=f[['z0','z1']].to_numpy()
    rotation=np.array([[0,-1],[1,0]]);y=7.3*z@rotation+np.array([10,20])
    scores=evaluate(f,ref,q,y[ref],y[q])
    assert scores.recall15.min()==1
    assert scores.abs_log_radius15.max()<1e-11
    np.testing.assert_allclose(scores.scale,1/7.3,atol=1e-12)
    ii,_=neighbors(z[ref],z[ref],15,np.arange(len(ref)))
    assert not (ii==np.arange(len(ref))[:,None]).any()

def test_false_shortcut_and_legitimate_junction():
    # Two branches sharing node1: points very close to junction are legitimate,
    # even when a tiny synthetic source radius would otherwise flag them.
    f=pd.DataFrame(dict(structure=[1,1],branch=[0,1],along=[2.99,.01],length=[3.,np.sqrt(8)],theta=[np.nan,np.nan],z0=[0,-.007],z1=[-2.01,-1.993]))
    _,wrong=shortcut_flags(f,f,np.array([[1],[0]]),np.array([.001,.001]))
    assert not wrong.any()
    f.loc[0,['along','z1']]=[0,-5];f.loc[1,['along','z0','z1']]=[np.sqrt(8),-2,0]
    _,wrong=shortcut_flags(f,f,np.array([[1],[0]]),np.array([.1,.1]));assert wrong.all()

def test_repulsion_resume_preserves_optimizer_and_teacher_rng():
    import copy
    import torch
    from umaping.config import RepulsionConfig
    from umaping.dynamics import ReferenceTrajectory
    from umaping.models.repulsion import RepulsionField
    from umaping.repulsion_estimators import TrainingQueries,UniformMC
    from umaping.training.flow import train_repulsion_field,RepulsionTrainState
    torch.set_num_threads(1)
    rng=np.random.default_rng(8);positions=rng.normal(size=(3,20,2)).astype(np.float32)
    trajectory=ReferenceTrajectory(np.array([0.,.5,1.]),positions)
    cfg=RepulsionConfig(hidden_dim=16,n_residual_blocks=1,time_embed_dim=4,steps=4,batch_size=8)
    torch.manual_seed(5);original=RepulsionField(embedding_dim=2,hidden_dim=16,n_residual_blocks=1,time_embed_dim=4)
    hp=copy.deepcopy(original.state_dict())
    def setup():
        model=copy.deepcopy(original);model.load_state_dict(hp)
        teacher=UniformMC(trajectory=trajectory,a=1.5,b=.9,seed=9,samples=8)
        sampler=TrainingQueries(trajectory,'cpu',8,.1,10)
        return model,teacher,sampler
    model,teacher,sampler=setup()
    full=train_repulsion_field(model,trajectory,1.5,.9,cfg,np.ones(20),torch.device('cpu'),teacher=teacher,query_sampler=sampler,progress=False)
    expected=copy.deepcopy(model.state_dict());model,teacher,sampler=setup();saved={}
    def checkpoint(state,m,opt):
        if state.step==2:saved.update(state_dict=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),teacher_rng=copy.deepcopy(teacher.rng.bit_generator.state),losses=state.losses.copy())
    train_repulsion_field(model,trajectory,1.5,.9,cfg,np.ones(20),torch.device('cpu'),teacher=teacher,query_sampler=sampler,progress=False,checkpoint_callback=checkpoint)
    model,teacher,sampler=setup();model.load_state_dict(saved['state_dict']);teacher.rng.bit_generator.state=saved['teacher_rng']
    resumed=train_repulsion_field(model,trajectory,1.5,.9,cfg,np.ones(20),torch.device('cpu'),teacher=teacher,query_sampler=sampler,progress=False,resume_state=RepulsionTrainState(step=2,losses=saved['losses']),optimizer_state=saved['optimizer'])
    assert full.losses==resumed.losses
    for name,value in model.state_dict().items():torch.testing.assert_close(value,expected[name],rtol=0,atol=0)
