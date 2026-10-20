from model import objectives
from .clip_model import (Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained,
                         build_CLIP_from_irra_checkpoint, convert_weights)
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from .scfp.feature_frequency import AdaptiveFeatureFrequency
from .scfp.semantic_slots import DualModalSemanticSlots
from .scfp.local_alignment import local_similarity, local_alignment_loss
from .scfp.diversity import slot_diversity


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        checkpoint = (getattr(args, 'backbone_checkpoint', '') or
                      getattr(args, 'resume_checkpoint', '') or getattr(args, 'init_checkpoint', ''))
        if checkpoint:
            self.base_model, base_cfg = build_CLIP_from_irra_checkpoint(checkpoint, args.img_size, args.stride_size)
        else:
            self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch, return_features=False):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        frequency = getattr(self.base_model.visual, 'scfp_frequency', None)
        if frequency is not None:
            frequency.capture = return_features
        image_feats, text_feats = self.base_model(images, caption_ids)
        self.scfp_features = {}
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.to(self.classifier.weight.dtype)).float()
            text_logits = self.classifier(t_feats.to(self.classifier.weight.dtype)).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if hasattr(self, 'scfp_slots'):
            features = self.scfp_slots(image_feats[:, 1:], text_feats, caption_ids)
            similarity = local_similarity(features['visual_slots'], features['text_slots'],
                                          features['slot_relevance'])
            ret['slot_loss'] = local_alignment_loss(similarity, batch['pids'], logit_scale) * self.args.scfp_slot_loss_weight
            ret['diversity_loss'] = (slot_diversity(features['visual_slots']) +
                                     slot_diversity(features['text_slots'])) * 0.5 * self.args.scfp_diversity_weight
            self.scfp_features = {k: v.detach() for k, v in features.items()}
            self.scfp_features['local_similarity'] = similarity.detach()
        if return_features:
            features_out = dict(features) if hasattr(self, 'scfp_slots') else {}
            frequency = getattr(self.base_model.visual, 'scfp_frequency', None)
            if frequency is not None:
                features_out.update(frequency.last_features)
            return ret, features_out
        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    if getattr(args, 'precision', 'fp16') == 'fp16':
        convert_weights(model)
    mode = getattr(args, 'scfp_mode', 'baseline')
    if mode not in ('baseline', 'slots', 'frequency', 'v2'):
        raise ValueError(f'Unknown SCFP mode: {mode}')
    model.scfp_mode = mode
    if mode in ('frequency', 'v2'):
        visual = model.base_model.visual
        if not hasattr(visual, 'transformer'):
            raise ValueError('SCFP requires a CLIP ViT backbone')
        layer = getattr(args, 'scfp_frequency_layer', 6)
        if not 1 <= layer < len(visual.transformer.resblocks):
            raise ValueError('Frequency layer must precede at least one remaining ViT block')
        visual.scfp_frequency_layer = layer
        visual.scfp_frequency = AdaptiveFeatureFrequency(
            visual.conv1.out_channels,
            getattr(args, 'scfp_frequency_num_bands', 4),
            getattr(args, 'scfp_frequency_hidden_dim', 64),
            getattr(args, 'scfp_frequency_temperature', 1.0),
            getattr(args, 'scfp_frequency_alpha_init', 0.01))
    if mode in ('slots', 'v2'):
        model.scfp_slots = DualModalSemanticSlots(
            model.embed_dim, getattr(args, 'scfp_num_slots', 4),
            getattr(args, 'scfp_slot_heads', 4), getattr(args, 'scfp_slot_dropout', 0.0),
            args.vocab_size)
    if getattr(args, 'freeze_backbone', False):
        for name, parameter in model.base_model.named_parameters():
            if 'scfp_frequency.' not in name:
                parameter.requires_grad_(False)
    return model
