"""Legal training interface. No task semantics or evaluation data access."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def state_hash(state):
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

def codes():
    return torch.tensor([[(n >> j) & 1 for j in range(4)] for n in range(16)], dtype=torch.float32)

class Sender(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,8))
    def logits(self, x, q):
        return self.net(torch.cat((x,F.one_hot(q.long(),2).float()),1))[:,:4]
    def message(self,x,q,ste=False):
        z=self.logits(x,q); h=(z>=0).float()
        if ste:
            p=z.sigmoid(); h=h.detach()-p.detach()+p
        return torch.cat((h,torch.zeros_like(h)),1)

class Receiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(22,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,4))
    def forward(self,m,q,r):
        mask=torch.zeros_like(m); mask[:,:4]=1
        return self.net(torch.cat((m,mask,F.one_hot(q.long(),2).float(),r),1))

def load_receiver(path):
    r=Receiver(); r.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['receiver'])
    r.eval()
    for p in r.parameters(): p.requires_grad_(False)
    return r

def initial_sender(init,cfg):
    torch.manual_seed(init+cfg['sender_seed_offset'])
    return Sender()

def legal_data(path):
    with np.load(path) as f:
        assert set(f.files)=={'x','q','r','y'}
        return {k:torch.tensor(f[k],dtype=torch.float32 if k!='q' else torch.int64) for k in f.files}

@torch.no_grad()
def candidate_losses(receiver,q,r,y,chunk=128):
    c=codes(); padded=torch.cat((c,torch.zeros_like(c)),1); rows=[]
    for start in range(0,len(q),chunk):
        b=len(q[start:start+chunk])
        logits=receiver(padded.repeat(b,1),q[start:start+chunk].repeat_interleave(16),r[start:start+chunk].repeat_interleave(16,0))
        losses=F.binary_cross_entropy_with_logits(logits,y[start:start+chunk].repeat_interleave(16,0),reduction='none').mean(1).reshape(b,16)
        rows.append(losses)
    result=torch.cat(rows)
    assert torch.isfinite(result).all()
    return result

def targets(losses,tau):
    c=codes(); hard=c[losses.argmin(1)]
    a=-losses/tau; w=(a-a.max(1,keepdim=True).values).softmax(1)
    return hard.detach(),(w@c).detach(),w.detach()

def objective(sender,receiver,d,arm,target=None):
    if arm=='S-STE':
        return F.binary_cross_entropy_with_logits(receiver(sender.message(d['x'],d['q'],True),d['q'],d['r']),d['y'])
    return F.binary_cross_entropy_with_logits(sender.logits(d['x'],d['q']),target)

@torch.no_grad()
def predict(sender,receiver,d):
    m=sender.message(d['x'],d['q']); l=receiver(m,d['q'],d['r'])
    return m,l,{'exact_accuracy':float(((l>=0)==d['y'].bool()).all(1).float().mean()),'bit_accuracy':float(((l>=0)==d['y'].bool()).float().mean()),'task_bce':float(F.binary_cross_entropy_with_logits(l,d['y']))}
