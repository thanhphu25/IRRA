import pytest
import torch

from model.clip_model import CLIP
from utils.options import get_args


@pytest.fixture
def make_model(monkeypatch):
    # Real IRRA, CLIP transformer, SDM/MLM/ID code; small random CLIP, no download.
    import model.build as build
    settings = dict(vocab_size=32, text_length=8)
    def tiny_clip(*args):
        return CLIP(embed_dim=64, image_resolution=(32, 16), vision_layers=1,
                    vision_width=64, vision_patch_size=8, stride_size=8,
                    context_length=settings['text_length'], vocab_size=settings['vocab_size'], transformer_width=64,
                    transformer_heads=1, transformer_layers=1), {'embed_dim': 64}
    monkeypatch.setattr(build, 'build_CLIP_from_openai_pretrained', tiny_clip)
    def factory(mode='baseline', **overrides):
        args = get_args(['--device', 'cpu', '--MLM', '--cmt_depth', '1',
                         '--vocab_size', '32', '--text_length', '8'])
        args.lgsp_mode = mode
        args.img_size = (32, 16)
        args.distributed = False
        for k, v in overrides.items():
            setattr(args, k, v)
        settings.update(vocab_size=args.vocab_size, text_length=args.text_length)
        torch.manual_seed(5)
        return build.build_model(args, num_classes=3)
    return factory


@pytest.fixture
def batch():
    torch.manual_seed(12)
    ids = torch.randint(1, 29, (3, 8))
    ids[:, -1] = 31
    labels = torch.zeros_like(ids)
    labels[:, 2] = ids[:, 2]
    masked = ids.clone()
    masked[:, 2] = 30
    return dict(images=torch.randn(3, 3, 32, 16), caption_ids=ids, pids=torch.arange(3),
                mlm_ids=masked, mlm_labels=labels)
