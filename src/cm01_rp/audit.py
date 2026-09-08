"""Independent scalar recomputation from archived response tensors."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def audit(output):
    summary=json.loads((output/'summary.json').read_text())
    config=json.loads((output/'config.lock.json').read_text())
    assert len(summary['models'])==6
    assert sorted(r['model'] for r in summary['models'])==[f'{s}_{i}_J-GROUP16' for s in (9001,9002,9003) for i in (9201,9202)]
    checks=[]
    for row in summary['models']:
        with np.load(output/f"{row['model']}_responses.npz",allow_pickle=False) as f:
            a={k:f[k] for k in f.files}
        p=1/(1+np.exp(-a['logits'].astype(float)))
        joint=np.empty((2,16,16,16))
        for q,r,m,y in itertools.product(range(2),range(16),range(16),range(16)):
            joint[q,r,m,y]=np.prod([p[q,r,m,b] if (y>>b)&1 else 1-p[q,r,m,b] for b in range(4)])
        np.testing.assert_allclose(joint,a['joint_probabilities'],atol=1e-14,rtol=1e-12)
        np.testing.assert_allclose(joint.sum(-1),1,atol=1e-12)
        for q in range(2):
            ctx=row['contexts'][q]
            counts=np.zeros((16,16),dtype=int)
            for r,m in itertools.product(range(16),repeat=2):
                answer=sum((int(a['logits'][q,r,m,b]>=0)<<b) for b in range(4))
                counts[m,(answer-r)%16]+=1
            assert counts.tolist()==ctx['coverage_counts']
            missing=[v for v in ctx['necessary_v'] if max(counts[:,v])<16]
            assert missing==ctx['missing_v']
            near=[]
            for i,j in itertools.combinations(range(16),2):
                tv=[]
                for r in range(16):
                    v=sum(abs(float(joint[q,r,i,y])-float(joint[q,r,j,y])) for y in range(16))/2
                    assert abs(v-a['tv'][q,r,i,j])<1e-12
                    tv.append(v)
                    left,right=joint[q,r,i],joint[q,r,j]
                    mid=(left+right)/2
                    js=.5*sum(float(x)*np.log2(float(x)/float(z)) for arr in (left,right) for x,z in zip(arr,mid) if x>0)
                    assert abs(js-a['js_bits'][q,r,i,j])<1e-12
                if sum(tv)/16<=config['mean_tv_max'] and max(tv)<=config['max_tv_max']:
                    near.append([i,j])
            usage=np.bincount(a['train_group_messages'][a['train_group_q']==q],minlength=16)
            assert usage.tolist()==ctx['usage']
            occupied=[[i,j] for i,j in near if usage[i] and usage[j]]
            assert near==ctx['near_pairs'] and occupied==ctx['occupied_near_pairs']
            assert bool(occupied and missing)==ctx['positive']
        assert row['positive']==any(c['positive'] for c in row['contexts'])
        checks.append({'model':row['model'],'passed':True})
    assert summary['positive_count']==sum(r['positive'] for r in summary['models'])
    report={'passed':True,'checks':checks,'scope':'independent product distributions, TV, JS, semantic coverage, occupancy and structural gates'}
    (output/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output)
