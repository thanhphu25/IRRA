"""Compare disabled-LGSP behavior to the audited Git baseline, without downloads.

Run on the full Git checkout: python -m experiments.lgsp_tbpr.audit_baseline
Uses small random CLIP and legacy FP16 paths on CPU, not measured TBPR scores.
"""
import subprocess
import types
import torch
from model.clip_model import CLIP
from utils.options import get_args
from utils.metrics import rank_full_chunked
import model.build as current

COMMIT = 'cfc23785545ead82dee2749fa0fc0ad948c96430'
def load_original(path, name):
    source = subprocess.check_output(['git', 'show', COMMIT + ':' + path], text=True)
    module = types.ModuleType(name)
    module.__package__ = 'model' if path.startswith('model/') else 'utils'
    exec(compile(source, path, 'exec'), module.__dict__)
    return module

legacy = load_original('model/build.py', 'model.legacy_build')
legacy_metrics = load_original('utils/metrics.py', 'utils.legacy_metrics')
def tiny_clip(*args):
    return CLIP(embed_dim=64, image_resolution=(32,16), vision_layers=1,
        vision_width=64, vision_patch_size=8, stride_size=8, context_length=8,
        vocab_size=32, transformer_width=64, transformer_heads=1, transformer_layers=1), {'embed_dim':64}
legacy.build_CLIP_from_openai_pretrained = current.build_CLIP_from_openai_pretrained = tiny_clip
args = get_args(['--cmt_depth', '1', '--vocab_size', '32', '--text_length', '8'])
args.img_size = (32,16)
torch.manual_seed(5); old = legacy.build_model(args,3).eval()
torch.manual_seed(5); new = current.build_model(args,3).eval()
assert set(old.state_dict()) == set(new.state_dict())
assert all(torch.equal(old.state_dict()[k],new.state_dict()[k]) for k in old.state_dict())
torch.manual_seed(12)
ids = torch.randint(1,29,(3,8)); ids[:,-1]=31
labels = torch.zeros_like(ids); labels[:,2]=ids[:,2]
masked = ids.clone(); masked[:,2]=30
batch = dict(images=torch.randn(3,3,32,16), caption_ids=ids, pids=torch.arange(3), mlm_ids=masked, mlm_labels=labels)
with torch.no_grad():
    a,b = old(batch),new(batch)
    assert set(a) == set(b)
    for k in a:
        torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    torch.testing.assert_close(old.encode_image(batch['images']),new.encode_image(batch['images']),rtol=0,atol=0)
    torch.testing.assert_close(old.encode_text(ids),new.encode_text(ids),rtol=0,atol=0)
print('Upstream baseline FP16: identical state keys, all state tensors, all SDM/MLM/ID outputs and image/text embeddings (bitwise).')
torch.manual_seed(7)
q,g = torch.randn(13,8),torch.randn(17,8)
qi,gi = torch.arange(13)%4,torch.arange(17)%4
expected = legacy_metrics.rank(q@g.T,qi,gi)[:3]
actual = rank_full_chunked(q,g,qi,gi,chunk_size=3)
for a,b in zip(expected,actual):
    torch.testing.assert_close(a,b)
print('Upstream full-matrix metrics vs chunked: numerically equivalent CMC/mAP/mINP.')
