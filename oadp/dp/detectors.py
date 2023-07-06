__all__ = [
    'GlobalHead',
    'OADP',
]

from typing import Any

import einops
import todd
import torch
from mmdet.core import BitmapMasks
from mmdet.models import DETECTORS, RPNHead, TwoStageDetector
from mmdet.models.utils.builder import LINEAR_LAYERS
from todd.distillers import SelfDistiller, Student
from todd.losses import LossRegistry as LR

from ..base import Globals
from .roi_heads import OADPRoIHead
from .utils import MultilabelTopKRecall

# from clip import clip
import torch.nn.functional as F
from .attention import ObjectAttention
device = "cuda" if torch.cuda.is_available() else "cpu"  # TODO 从配置中读取设备


class GlobalHead(todd.Module):

    def __init__(
        self,
        *args,
        topk: int,
        classifier: todd.Config,
        loss: todd.Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._multilabel_topk_recall = MultilabelTopKRecall(k=topk)
        self._classifier = LINEAR_LAYERS.build(classifier)
        self._loss = LR.build(loss)

    def forward(self, feats: tuple[torch.Tensor, ...]) -> torch.Tensor:
        feat = einops.reduce(feats[-1], 'b c h w -> b c', reduction='mean')
        return self._classifier(feat)

    def forward_train(
        self,
        *args,
        labels: list[torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        logits = self.forward(*args, **kwargs)
        targets = logits.new_zeros(
            logits.shape[0],
            Globals.categories.num_all,
            dtype=torch.bool,
        )
        for i, label in enumerate(labels):
            targets[i, label] = True
        return dict(
            loss_global=self._loss(logits.sigmoid(), targets),
            recall_global=self._multilabel_topk_recall(logits, targets),
        )


@DETECTORS.register_module()
class OADP(TwoStageDetector, Student[SelfDistiller]):
    rpn_head: RPNHead
    roi_head: OADPRoIHead

    def __init__(
        self,
        *args,
        global_head: todd.Config,
        distiller: todd.Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._global_head = GlobalHead(**global_head)
        distiller.setdefault('type', 'SelfDistiller')
        Student.__init__(self, distiller)
        self.object_attn = ObjectAttention(d_model=512, n_head=8, device=device)

    @property
    def num_classes(self) -> int:
        return Globals.categories.num_all

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: list[dict[str, Any]],
        gt_bboxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
        clip_global: torch.Tensor,
        clip_blocks: list[torch.Tensor],
        block_bboxes: list[torch.Tensor],
        block_labels: list[torch.Tensor],
        clip_objects: list[torch.Tensor],
        object_bboxes: list[torch.Tensor],
        gt_masks: list[BitmapMasks] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        Globals.training = True
        feats = self.extract_feat(img)
        global_losses = self._global_head.forward_train(
            feats,
            labels=gt_labels,
        )

        rpn_losses, proposals = self.rpn_head.forward_train(
            feats,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=None,
            proposal_cfg=self.train_cfg.rpn_proposal,
            **kwargs,
        )

        roi_losses = self.roi_head.forward_train(
            feats,
            img_metas,
            proposals,
            gt_bboxes,
            gt_labels,
            None,
            gt_masks,
            **kwargs,
        )
        block_losses = self.roi_head.block_forward_train(
            feats,
            block_bboxes,
            block_labels,
        )
        self.roi_head.object_forward_train(feats, object_bboxes)

        clip_objects = self.reweight_objects(clip_global, clip_objects)

        custom_tensors = dict(
            clip_global=clip_global.float(),
            clip_blocks=torch.cat(clip_blocks).float(),
            clip_objects=torch.cat(clip_objects).float(),
        )
        distill_losses = self.distiller(custom_tensors)
        self.distiller.reset()
        self.distiller.step()

        return {
            **global_losses,
            **rpn_losses,
            **roi_losses,
            **block_losses,
            **distill_losses,
        }

    def simple_test(self, *args, **kwargs):
        Globals.training = False
        return super().simple_test(*args, **kwargs)
    
    def reweight_objects(self, clip_global:torch.Tensor, clip_objects:list[torch.Tensor]):
        batch_size = clip_global.shape[0]
        new_objects = []
        for i in range(batch_size):
            clip_global_per_image = clip_global[i]
            clip_objects_per_image = clip_objects[i]

            # 将 shape 转换到适合 MultiHeadAttention 的结构
            clip_objects_per_image = torch.unsqueeze(clip_objects_per_image, 0)
            clip_global_per_image = torch.unsqueeze(torch.unsqueeze(clip_global_per_image, 0), 0)

            q = clip_objects_per_image.float()  # torch.Size([1, C, 512]), C 个 proposals embeddings
            k = clip_global_per_image.float()   # torch.Size([1, 1, 512]) , 1 个 global embeddings
            v = clip_global_per_image.float()   # torch.Size([1, 1, 512]) 

            out = self.object_attn(q, k, v)    # torch.Size([1, C, 512])
            out = torch.squeeze(out, 0)
            new_objects.append(out)

        return new_objects
