import torch
import time
import torch.nn as nn
import sys
import logging as lg
import random as r
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision
import torch.cuda.amp as amp
import random
import wandb
import matplotlib.pyplot as plt
import importlib

# 修改说明（按模块划分）：
# - 文件顶部：说明引入 timm 预训练 ViT 作为基础教师，并允许从最后 CLS 或中间层提取特征，作为 MoE 门控的先验输入。
# - MoEGatingTeacher 类：新增轻量 MoE 路由模块。其作用是读取 ViT 的 CLS/中间 token，经 LazyLinear 产生专家权重，支持 top-k 稀疏路由，然后对各 EMA 教师 logits 做加权融合；同时返回 ViT 自身 logits，用于和 MoE 融合一起蒸馏。
# - ER_EMALearner.__init__：在原有 EMA 教师之外初始化 timm 教师与 MoE 教师，便于后续蒸馏。
# - init_timm_teacher：封装 timm 预训练 ViT 的加载与冻结，仅做推理不训练。
# - init_moe_teacher：将 timm ViT 与 EMA 教师拼装为 MoE 路由网络，未开启开关或缺少 timm/EMA 时返回 None。
# - train：训练时同时收集 EMA、timm、MoE 的 logits 参与蒸馏，MoE 会把基准 ViT logits 与门控后的 EMA logits 一并加入，学生平均对齐这些教师信号。

from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE

from src.learners.base import BaseLearner
from src.learners.baselines.er import ERLearner
from src.utils.losses import WKDLoss 
from src.models.resnet import ResNet18
from src.utils import name_match
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device, filter_labels
from src.utils.augment import MixupAdaptative, ZetaMixup

device = get_device()

scaler = amp.GradScaler()

LR_MIN = 5e-4
LR_MAX = 5e-2

# 优先尝试的 ImageNet21K 预训练 ViT 模型名列表（均在 timm 中常见）。
IN21K_PREFERRED = [
    "vit_tiny_patch16_224_in21k",
    "vit_small_patch32_224_in21k",
    "vit_small_patch16_224_in21k",
    "vit_base_patch32_224_in21k",
    "vit_base_patch16_224_in21k",
    "vit_base_patch8_224_in21k",
    "vit_large_patch32_224_in21k",
    "vit_large_patch16_224_in21k",
    "vit_huge_patch14_224_in21k",
]

# 来自 Google 提供的 ImageNet21K 预训练权重（npz/pth），可直接作为 timm teacher 的下载源。
IN21K_FALLBACK_URLS = {
    "vit_tiny_patch16_224_in21k": "https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
    "vit_small_patch32_224_in21k": "https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    "vit_small_patch16_224_in21k": "https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    "vit_base_patch32_224_in21k": "https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz",
    "vit_base_patch16_224_in21k": "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    "vit_base_patch8_224_in21k": "https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    "vit_large_patch32_224_in21k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth",
    "vit_large_patch16_224_in21k": "https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz",
    "vit_huge_patch14_224_in21k": "https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz",
}


def _extract_default_cfg_url(cfg_entry):
    """兼容 timm default_cfgs 中不同类型的配置对象，提取其 url 字段。"""

    if cfg_entry is None:
        return None

    if isinstance(cfg_entry, dict):
        return cfg_entry.get("url") or cfg_entry.get("hf_hub_id")

    # PretrainedCfg 对象既支持属性访问也支持类似 dict 的 get
    url = getattr(cfg_entry, "url", None)
    if url:
        return url

    get_fn = getattr(cfg_entry, "get", None)
    if callable(get_fn):
        try:
            return get_fn("url", None) or get_fn("hf_hub_id", None)
        except TypeError:
            pass
    return None


class MoEGatingTeacher(nn.Module):
    """
    A lightweight routing MoE that uses frozen ViT features as gating signals for EMA experts.
    """

    def __init__(self, base_teacher, experts, use_intermediate=False, intermediate_index=-1, top_k=1):
        super().__init__()
        # 说明：记录预训练 ViT（base_teacher）与一组 EMA 专家（experts），后续门控时要用到它们。
        self.base_teacher = base_teacher
        self.experts = experts
        self.expert_keys = list(experts.keys())
        self.use_intermediate = use_intermediate
        self.intermediate_index = intermediate_index
        self.top_k = top_k

        # 说明：懒初始化全连接层，输入维度由 ViT token 推理得到，输出等于专家数；权重冻结，仅做路由估计。
        self.gating = nn.LazyLinear(len(self.experts), bias=False)
        self.gating.weight.requires_grad_(False)

    @torch.no_grad()
    def extract_token(self, x):
        """
        Extract the CLS token (or intermediate block output) from the ViT teacher.
        """
        # 说明：支持从中间层或最终 CLS 取特征，保证 MoE 门控可用 ViT 的不同深度信息。
        if self.use_intermediate and hasattr(self.base_teacher, "get_intermediate_layers"):
            n_layers = max(1, abs(self.intermediate_index))
            intermediates = self.base_teacher.get_intermediate_layers(
                x, n=n_layers, reshape=False, return_class_token=True
            )
            picked = intermediates[self.intermediate_index] if intermediates else None
            token = picked[:, 0] if picked is not None else self.base_teacher.forward_features(x)
        elif hasattr(self.base_teacher, "forward_features"):
            feats = self.base_teacher.forward_features(x)
            if isinstance(feats, (list, tuple)):
                feats = feats[self.intermediate_index]
            elif isinstance(feats, dict):
                feats = feats.get("x", feats.get("out", feats))
            token = feats[:, 0] if feats.dim() == 3 else feats
        else:
            token = self.base_teacher(x)
        return token

    def logits(self, x):
        # 说明：推理阶段仅前向，不更新 ViT 与 EMA；返回 MoE 聚合 logits、基准 ViT logits 以及本次被路由的专家编号，供蒸馏融合和 EMA 更新使用。
        with torch.no_grad():
            base_logits = self.base_teacher(x)
            token = self.extract_token(x)

        gating_weights = torch.softmax(self.gating(token), dim=-1)

        # 说明：如开启 top-k，保留最重要的专家权重，实现稀疏路由；同时记录被选中的专家索引。
        selected_indices = None
        if self.top_k > 0 and self.top_k < gating_weights.shape[1]:
            topk_vals, topk_idx = torch.topk(gating_weights, self.top_k, dim=1)
            sparse_weights = torch.zeros_like(gating_weights)
            sparse_weights.scatter_(1, topk_idx, topk_vals)
            gating_weights = sparse_weights
            selected_indices = topk_idx

        expert_logits = []
        with torch.no_grad():
            for expert in self.experts.values():
                # 说明：支持模型暴露 logits 接口或直接可调用，便于兼容现有 EMA 教师。
                if hasattr(expert, "logits"):
                    expert_logits.append(expert.logits(x))
                else:
                    expert_logits.append(expert(x))

        stacked_logits = torch.stack(expert_logits, dim=1)
        moe_logits = torch.einsum("be,ben->bn", gating_weights, stacked_logits)

        # 说明：若未使用稀疏路由，则默认所有专家都被视作参与，本次更新会同步所有 EMA；若使用 top-k，则仅更新被选中编号对应的 EMA。
        if selected_indices is None:
            selected_keys = self.expert_keys
        else:
            unique_idx = torch.unique(selected_indices).cpu().tolist()
            selected_keys = [self.expert_keys[idx] for idx in unique_idx]

        return moe_logits, base_logits, selected_keys

class ER_EMALearner(ERLearner):
    def __init__(self, args):
        super().__init__(args)
        self.wkdloss = WKDLoss(
            temperature=self.params.kd_temperature,
            use_wandb= not self.params.no_wandb,
            alpha_kd=self.params.alpha_kd
        )

        self.timm_teacher = self.init_timm_teacher()
        self.moe_teacher = self.init_moe_teacher()

        self.classes_seen_so_far = torch.LongTensor(size=(0,)).to(device)
        
        self.ema_models = {}
        self.ema_alphas = {}
        if self.params.alpha_min is None or self.params.alpha_max is None:
            # Manually set 5 teachers
            self.ema_models[0] = deepcopy(self.model)
            self.ema_models[1] = deepcopy(self.model)
            self.ema_models[2] = deepcopy(self.model)
            self.ema_models[3] = deepcopy(self.model)
            
            self.ema_alphas[0] = self.params.ema_alpha1
            self.ema_alphas[1] = self.params.ema_alpha2
            self.ema_alphas[2] = self.params.ema_alpha3
            self.ema_alphas[3] = self.params.ema_alpha4
        else:
            # Automatically set a specific number of teachers
            for i in range(self.params.n_teacher):
                self.ema_models[i] = deepcopy(self.model)
                self.ema_alphas[i] = 10**(
                    np.log10(self.params.alpha_min) + 
                    (np.log10(self.params.alpha_max) - np.log10(self.params.alpha_min))*i/(max(1,self.params.n_teacher - 1))
                    )
        print(self.ema_alphas)

        self.update_ema(init=True)
        
        self.previous_model = None
        if self.params.measure_drift >= 0:
            self.drift = []
            self.previous_model = None

    def get_teacher_logits(self, teacher, combined_aug, combined_x):
        if hasattr(teacher, "logits"):
            logits_aug = teacher.logits(combined_aug)
            logits_raw = teacher.logits(combined_x) if not self.params.no_aug else None
        else:
            logits_aug = teacher(combined_aug)
            logits_raw = teacher(combined_x) if not self.params.no_aug else None
        return logits_aug, logits_raw

    def init_timm_teacher(self):
        if not getattr(self.params, "timm_teacher", False):
            return None

        timm_spec = importlib.util.find_spec("timm")
        if timm_spec is None:
            raise ImportError("timm is required for loading a pretrained teacher. Please install timm first.")

        import timm

        teacher_name = self.resolve_timm_teacher_name(timm)
        teacher = self.build_timm_teacher(timm, teacher_name)
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        teacher.to(device)
        return teacher

    def resolve_timm_teacher_name(self, timm):
        """
        只允许使用 ImageNet21K 预训练的 ViT：
        - 若用户指定了 timm_teacher_name 且可用，则直接使用（推荐 *_in21k 变体）。
        - 否则按 IN21K_PREFERRED 顺序选择本地可用的 in21k 模型。
        - 若本地 timm 不包含任何 in21k 变体，则直接报错，提示安装带有 in21k 权重的 timm 包。
        """

        requested = getattr(self.params, "timm_teacher_name", None)
        available = set(timm.list_models(pretrained=True))

        if requested:
            if requested in available:
                if requested not in IN21K_FALLBACK_URLS:
                    lg.warning("建议使用 *_in21k 变体以匹配 Google 权重；当前模型将尝试使用提供的 URL 或 default_cfgs 下载。")
                return requested
            lg.warning("用户指定的 timm 教师在当前环境不可用，尝试自动选择 in21k 变体。")

        for name in IN21K_PREFERRED:
            if name in available:
                lg.warning(
                    f"自动选择可用的 ImageNet21K 预训练模型：{name}。如需固定请显式传入 --timm-teacher-name 并保证本地可用。"
                )
                return name

        raise RuntimeError(
            "本地 timm 未找到任何 ImageNet21K ViT（*_in21k）。请安装包含 in21k 变体的 timm 或手动指定 --timm-teacher-name 并准备对应 URL。"
        )

    def build_timm_teacher(self, timm, teacher_name):
        """
        仅加载 Google 提供的 ImageNet21K 权重：
        - 优先使用用户指定的 --timm-pretrained-url（npz/pth），直接通过 pretrained_cfg_overlay 下载。
        - 若未显式给定 URL，则从 timm default_cfgs 解析或内置回退表寻找 Google in21k 链接，同样用 overlay 拉取。
        - 仅当用户提供了 timm_pretrained_cfg 时才尝试对应标签，否则默认不再走 timm 的 1K 权重。
        """

        user_cfg = getattr(self.params, "timm_pretrained_cfg", None)
        user_url = getattr(self.params, "timm_pretrained_url", None)

        vit_default_cfgs = getattr(timm.models.vision_transformer, "default_cfgs", {})
        cfg_entry = vit_default_cfgs.get(teacher_name)
        default_url = _extract_default_cfg_url(cfg_entry)

        chosen_url = user_url or default_url or IN21K_FALLBACK_URLS.get(teacher_name)
        if not chosen_url:
            raise RuntimeError(
                "未找到可用的 Google ImageNet21K 权重 URL。请通过 --timm-pretrained-url 提供 npz/pth 直链，或选择包含 in21k 权重的 timm 模型。"
            )

        overlay = {"url": chosen_url, "file": chosen_url}
        lg.info(f"将通过 URL 加载 ImageNet21K 权重：{chosen_url}")

        pretrained_cfg = user_cfg if user_cfg else None
        try:
            return timm.create_model(
                teacher_name,
                pretrained=True,
                num_classes=self.params.n_classes,
                pretrained_cfg=pretrained_cfg,
                pretrained_cfg_overlay=overlay,
            )
        except RuntimeError as e:
            lg.warning(f"timm 按 URL 加载 {teacher_name} 失败，错误：{e}")
            raise
    def init_moe_teacher(self):
        """
        Initialize a routing MoE teacher that uses ViT features from the timm teacher
        to gate EMA experts. Returns None if either timm teacher or EMA experts are
        unavailable, or if the user did not request MoE routing.
        """
        if not getattr(self.params, "moe_teacher", False):
            return None

        if self.timm_teacher is None:
            lg.warning("MoE teacher requested but no timm teacher is available; disabling MoE routing.")
            return None

        return MoEGatingTeacher(
            base_teacher=self.timm_teacher,
            experts=self.ema_models,
            use_intermediate=self.params.moe_use_intermediate,
            intermediate_index=self.params.moe_intermediate_index,
            top_k=self.params.moe_top_k,
        ).to(device)
            
    # @profile
    def train(self, dataloader, **kwargs):
        task_name = kwargs.get("task_name", "Unknown")
        task_id = kwargs.get('task_id', None)
        self.model = self.model.train()
        
        for j, batch in enumerate(dataloader):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                # Stream batch
                batch_x, batch_y = batch[0], batch[1]
                self.stream_idx += len(batch_x)
                # self.ema_model1.train()
                
                for _ in range(self.params.mem_iters):
                    # Iteration over memory + stream
                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    
                    if mem_x.size(0) > 0:
                        combined_x = torch.cat([mem_x, batch_x]).to(device)
                        combined_y = torch.cat([mem_y, batch_y]).to(device)
                        
                        # Augment
                        combined_aug = self.transform_train(combined_x)
                        
                        # logits
                        logits_stu = self.model.logits(combined_aug)
                        logits_stu_raw = self.model.logits(combined_x)

                        loss_dist = 0
                        teacher_pairs = []

                        # 说明：收集 EMA 教师的 logits，后面蒸馏时会与 MoE / timm 教师一起平均。
                        for teacher in self.ema_models.values():
                            logits_tea_aug, logits_tea_raw = self.get_teacher_logits(teacher, combined_aug, combined_x)
                            teacher_pairs.append((logits_tea_aug, logits_tea_raw))

                        selected_keys = None

                        if self.moe_teacher is not None:
                            # 说明：MoE 同时返回路由融合的 logits（moe_*）和基准 ViT logits（base_*），二者都参与蒸馏。
                            moe_aug, base_aug, selected_keys = self.moe_teacher.logits(combined_aug)
                            moe_raw, base_raw = (None, None)
                            if not self.params.no_aug:
                                moe_raw, base_raw, _ = self.moe_teacher.logits(combined_x)
                            # 说明：将 ViT 与 MoE 的 logits 按 β 与 1-β 融合，避免简单平均造成权重模糊。
                            beta = getattr(self.params, "moe_beta", 0.5)
                            fused_aug = beta * moe_aug + (1 - beta) * base_aug
                            fused_raw = None
                            if not self.params.no_aug:
                                fused_raw = beta * moe_raw + (1 - beta) * base_raw

                            teacher_pairs.append((fused_aug, fused_raw))
                        elif self.timm_teacher is not None:
                            # 说明：未开启 MoE 时仍可单独使用 timm 预训练 ViT 作为教师。
                            logits_tea_aug, logits_tea_raw = self.get_teacher_logits(self.timm_teacher, combined_aug, combined_x)
                            teacher_pairs.append((logits_tea_aug, logits_tea_raw))

                        # 说明：对所有教师的蒸馏损失求平均，确保 EMA、ViT、MoE 信号权重一致。
                        for logits_tea_aug, logits_tea_raw in teacher_pairs:
                            if self.params.no_aug or logits_tea_raw is None:
                                loss_dist += self.wkdloss(logits_tea_aug.detach(), logits_stu)
                            else:
                                loss_dist += (
                                    self.wkdloss(logits_tea_aug.detach(), logits_stu)
                                    + self.wkdloss(logits_tea_raw.detach(), logits_stu_raw)
                                ) / 2
                        loss_dist = loss_dist / len(teacher_pairs)
                        
                        loss_ce = self.criterion(logits_stu, combined_y.long())
                        loss = self.params.kd_lambda*loss_dist + loss_ce
                            
                        loss = loss.mean()

                        # Backprop
                        self.loss = loss.item()
                        
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                            scaler.scale(loss).backward()
                            scaler.step(self.optim)
                            scaler.update()
                        self.update_ema(selected_keys=selected_keys)
                        if self.params.annealing:
                            self.scheduler.step()
                        self.optim.zero_grad()
                        
                        if self.params.measure_drift >=0 and task_id > 0:
                            self.measure_drift(task_id)
                        
                        if not self.params.no_wandb:
                            wandb.log({
                                "loss_dist": loss_dist.item(),
                                "loss": loss.item()
                            })
                        print(f"Phase: {task_name}  Loss:{loss.item():.3f}  Loss dist:{loss_dist.item():.3f} batch {j}", end="\r")
                self.buffer.update(imgs=batch_x, labels=batch_y)
                if (j == (len(dataloader) - 1)) and (j > 0):
                    if self.params.tsne and task_id == 4:
                        self.tsne()
                    print(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s",
                        end="\r"
                    )
                    self.save(model_name=f"ckpt_{task_name}.pth")
    
    def train_uni(self, dataloader, **kwargs):
        raise NotImplementedError

    def update_ema(self, init=False, selected_keys=None):
        """
        Update the Exponential Moving Average (EMA) of the group of pytorch models.
        说明：若 selected_keys 不为空，则仅对路由选中的专家进行 EMA 更新，未被选中者保持不变。
        """
        keys_to_update = self.ema_models.keys() if selected_keys is None else selected_keys

        for key in keys_to_update:
            ema_model = self.ema_models[key]
            alpha = self.ema_alphas[key]
            for param, ema_param in zip(self.model.parameters(), ema_model.parameters()):
                p = deepcopy(param.data.detach())

                if init:
                    ema_param.data.mul_(0).add_(p * alpha / (1 - alpha ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))
                else:
                    ema_param.data.mul_(1 - alpha).add_(p * alpha / (1 - alpha ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))

    def encode(self, dataloader, model_tag=0, nbatches=-1, **kwargs):
        self.init_agg_model()
        if not self.params.drop_fc:
            i = 0
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(self.device)

                    logits = self.model_agg.logits(self.transform_test(inputs))
                    preds = nn.Softmax(dim=1)(logits).argmax(dim=1)
                    
                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_preds = preds.cpu().numpy()
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_preds = np.hstack([all_preds, preds.cpu().numpy()])
                    i += 1
            
            return all_preds, all_labels
        else:
            i = 0
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(self.device)
                    features, _ = self.model_agg(self.transform_test(inputs))
                    
                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_feat = features.cpu().numpy()
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_feat = np.vstack([all_feat, features.cpu().numpy()])
                    i += 1
            return all_feat, all_labels
    
    def init_agg_model(self):
        if self.params.eval_teacher:
            self.model_agg = deepcopy(list(self.ema_models.values())[0])
        else:
            self.model_agg = deepcopy(self.model)
            if not self.params.no_avg:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # infer with model_agg as average of all the ema models
                    with torch.no_grad():
                        for teacher in self.ema_models.values():
                            for param_agg, teacher_param in zip(self.model_agg.parameters(), teacher.parameters()):
                                param_agg.add_(teacher_param.detach())
                        for param_agg in self.model_agg.parameters():
                            param_agg.mul_(1/(len(self.ema_models) + 1))
        self.model_agg.eval()
    
    def get_mem_rep_labels(self, eval=True, use_proj=False):
        """Compute every representation -labels pairs from memory
        Args:
            eval (bool, optional): Whether to turn the mdoel in evaluation mode. Defaults to True.
        Returns:
            representation - labels pairs
        """
        self.init_agg_model()
        mem_imgs, mem_labels = self.buffer.get_all()
        batch_s = 10
        n_batch = len(mem_imgs) // batch_s
        all_reps = []
        for i in range(n_batch):
            mem_imgs_b = mem_imgs[i*batch_s:(i+1)*batch_s].to(self.device)
            mem_imgs_b = self.transform_test(mem_imgs_b)
            if use_proj:
                _, mem_representations_b = self.model_agg(mem_imgs_b)
            else:
                mem_representations_b, _ = self.model_agg(mem_imgs_b)
            all_reps.append(mem_representations_b)
        mem_representations = torch.cat(all_reps, dim=0)
        return mem_representations, mem_labels