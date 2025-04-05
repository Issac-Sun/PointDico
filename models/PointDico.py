import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from utils import misc
from .mask_encoder import Mask_Encoder, Group, Encoder, TransformerEncoder
from .generator import CPDM, H2Net
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from models.CrossModal import TextTransformer as TextEncoder
from models.CrossModal import VisionTransformer as ImageEncoder

'''We fuse the text and image features with the point cloud features.
We use the cross attention to fuse the features.'''

@MODELS.register_module()
class PointDif(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointDif] ', logger='PointDif')
        self.config = config
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.trans_dim = config.encoder_config.trans_dim
        self.mask_encoder = Mask_Encoder(config)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.drop_path_rate = config.encoder_config.drop_path_rate

        self.encoder_dims = config.encoder_config.encoder_dims
        self.cond_dims = config.generator_config.cond_dims
        self.ca_net = H2Net(self.encoder_dims, self.cond_dims)
        self.point_diffusion = CPDM(config)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        print_log(f'[PointDif] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='PointDif')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )
        trunc_normal_(self.mask_token, std=.02)

        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)
        # cross model contrastive
        self.csc_loss = torch.nn.SmoothL1Loss()
        self.csc_img = True if config.img_encoder else False
        self.csc_text = True if config.text_encoder else False
        if self.csc_img:
            self.img_encoder = ImageEncoder(config)
            for p in self.img_encoder.parameters():
                p.requires_grad = False
            self.img_proj = nn.Linear(self.trans_dim, self.img_encoder.output_dim)
            # self.img_proj = hypnn.HypLinear(self.trans_dim, self.img_encoder.output_dim,c=0.1)
            self.img_proj.apply(self._init_weights)

        if self.csc_text:
            self.text_encoder = TextEncoder(config)
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_proj = nn.Linear(self.trans_dim, self.text_encoder.embed_dim)
            # self.text_proj = hypnn.HypLinear(self.trans_dim, self.text_encoder.embed_dim,c=0.1)

            self.text_proj.apply(self._init_weights)

        # single modal contrastive
        self.smc = config.self_contrastive
        if self.smc:
            self.cls_proj = nn.Sequential(
                nn.Linear(self.trans_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128)
            )
            self.cls_proj.apply(self._init_weights)
            self.contrastive_head = ContrastiveHead(temperature=0.1)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.02, 0.01)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, pts,img, text, noaug=False, vis=False, **kwargs):
        losses = {}
        B, _, _ = pts.shape
        # get patch
        neighborhood, center = self.group_divider(pts)
        # mask and encoder
        cls_token, img_token, text_token, x_vis, mask,encoder_token= self.mask_encoder(neighborhood, center)
        _, N, _ = (center[mask].reshape(B, -1, 3)).shape
        # learnable masked token
        mask_token = self.mask_token.expand(B, N, -1)
        encoder_token[mask] = mask_token.reshape(-1, self.trans_dim)
        point_condition = self.ca_net(encoder_token)

        losses['dif'] = self.point_diffusion.get_loss(pts, point_condition)
        if self.csc_img:
            img_feature = self.img_encoder(img)
            img_token = self.img_proj(img_token)
            losses['csc_img'] = self.csc_loss(img_feature, img_token).mean()

        if self.csc_text:
            text_feature = self.text_encoder(text)
            text_token = self.text_proj(text_token)
            losses['csc_text'] = self.csc_loss(text_feature, text_token).mean()

        if self.smc:
            cls_proj = self.cls_proj(cls_token)
            cls_proj = nn.functional.normalize(cls_proj, dim=1)
            similarity = torch.matmul(cls_proj, cls_proj.permute(1, 0))

            select = torch.zeros([B, B], dtype=torch.uint8).to(similarity.device)
            for i in range(B):
                for j in range(B):
                    if text[i] == text[j]:
                        select[i, j] = 1
                        
            # Follow RECON paper, we donot use the SMC loss in the paper
            # losses['smc'] = loss.mean()
            
            losses['smc'] = self.contrastive_head(similarity, select)
        loss = sum(losses.values())
        if vis:  # visualization
            noise_points, recon_points = self.point_diffusion.sample(1024, point_condition)
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - N), -1, 3)
            vis_points = vis_points + center[~mask].unsqueeze(1)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            mask_center = misc.fps(pts, 256)
            vis_points = vis_points.reshape(-1, 3).unsqueeze(0)
            return noise_points, recon_points, vis_points, mask_center
        else:
            return loss


# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.img_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.text_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.img_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.text_pos = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.cls_pos, std=.02)

        trunc_normal_(self.img_token, std=.02)
        trunc_normal_(self.text_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.img_pos, std=.02)
        trunc_normal_(self.text_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.mask_encoder.", ""): v for k, v in ckpt['pointdif'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('pointdif'):
                    base_ckpt[k[len('pointdif.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        img_token = self.img_token.expand(group_input_tokens.size(0), -1, -1)
        img_pos = self.img_pos.expand(group_input_tokens.size(0), -1, -1)
        text_token = self.text_token.expand(group_input_tokens.size(0), -1, -1)
        text_pos = self.text_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens,img_token, text_token,  group_input_tokens), dim=1)

        pos = torch.cat((cls_pos,img_pos, text_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        # A.max(1)

        concat_f = torch.cat([x[:, 0], x[:, 1], x[:, 2], x[:, 3:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret