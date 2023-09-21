# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import DistAlignQueueHook, FixedThresholdingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, concat_all_gather

# pivotmatch11 + self contrast

class PivotMatch_Net(nn.Module):
    def __init__(self, base, proj_size=128):
        super(PivotMatch_Net, self).__init__()
        self.backbone = base
        self.num_features = base.num_features
        
        self.mlp_proj = nn.Sequential(*[
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(inplace=False),
            nn.Linear(self.num_features, proj_size)
        ])
        
    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out
    
    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        feat_proj = self.l2norm(self.mlp_proj(feat))
        return {'logits':logits, 'feat':feat_proj}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher

@ALGORITHMS.register('pivotmatch12')
class PivotMatch12(AlgorithmBase):
    """
    PivotMatch algorithm  
    Reference implementation 

    Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - T (`float`):
            Temperature for pseudo-label sharpening
        - p_cutoff(`float`):
            Confidence threshold for generating pseudo-labels
        - hard_label (`bool`, *optional*, default to `False`):
            If True, targets have [Batch size] shape with int values. If False, the target is vector
        - K (`int`, *optional*, default to 128):
            Length of the memory bank to store class probabilities and embeddings of the past weakly augmented samples
        - alpha (`float`, *optional*, default to 0.9):
            Weight for a smoothness constraint which encourages taking a similar value as its nearby samples’ class probabilities
        - da_len (`int`, *optional*, default to 256):
            Length of the memory bank for distribution alignment.
        - in_loss_ratio (`float`, *optional*, default to 1.0):
            Loss weight for pivotmatch feature loss
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # pivotmatch specified arguments
        # adjust k 
        self.lambda_c = args.contrast_loss_ratio
        self.use_ema_teacher = True
        if args.dataset in ['cifar10', 'cifar100', 'svhn', 'superks', 'tissuemnist', 'eurosat', 'superbks', 'esc50', 'gtzan', 'urbansound8k', 'aclImdb', 'ag_news', 'dbpedia']:
            self.use_ema_teacher = False

        # args.K = args.lb_dest_len
        self.init(T=args.T, p_cutoff=args.p_cutoff, proj_size=args.proj_size, hard_label=args.hard_label, 
                  queue_batch=args.queue_batch, alpha=args.alpha)
    
    def init(self, T, p_cutoff, proj_size, hard_label, queue_batch, alpha):
        self.T = T 
        self.p_cutoff = p_cutoff
        self.hard_label = hard_label 
        self.proj_size = proj_size 
        self.queue_batch = queue_batch
        self.alpha = alpha

        # memory bank
        self.queue_size = int(queue_batch * (self.args.uratio + 1) * self.args.batch_size)
        self.queue_ptr = 0
        self.queue_feats = torch.zeros(self.queue_size, self.args.proj_size).cuda(self.gpu)
        self.queue_probs = torch.zeros(self.queue_size, self.args.num_classes).cuda(self.gpu)
        self.queue_conf_set = torch.zeros(self.queue_size).cuda(self.gpu)

    def set_hooks(self):
        # self.register_hook(DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),  "DistAlignHook")
        # self.register_hook(DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),  "ClusterAlignHook")
        self.register_hook(DistAlignEMAHook(num_classes=self.num_classes, p_target_type='uniform'),  "DistAlignHook")
        self.register_hook(DistAlignEMAHook(num_classes=self.num_classes, p_target_type='uniform'),  "ClusterAlignHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook") 
        super().set_hooks()

    def set_model(self): 
        model = super().set_model()
        model = PivotMatch_Net(model, proj_size=self.args.proj_size)
        return model
    
    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = PivotMatch_Net(ema_model, proj_size=self.args.proj_size)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model    

    @torch.no_grad()
    def update_queue(self, feats, probs, mask):
        if self.distributed and self.world_size > 1:
            feats = concat_all_gather(feats)
            probs = concat_all_gather(probs)
            conf_set = concat_all_gather(mask)
        # update memory bank
        length = feats.shape[0]
        self.queue_feats[self.queue_ptr:self.queue_ptr + length, :] = feats
        self.queue_probs[self.queue_ptr:self.queue_ptr + length, :] = probs    
        self.queue_conf_set[self.queue_ptr:self.queue_ptr + length] = mask    
        self.queue_ptr = (self.queue_ptr + length) % self.queue_size
        
    @torch.no_grad()
    def get_lbs_and_masks(self, probs):
        max_probs, lbs_u_guess = torch.max(probs, dim=1)
        mask = max_probs.ge(self.p_cutoff).float()
        return lbs_u_guess, max_probs, mask

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, y_ulb):
        x_ulb_s_0 = x_ulb_s
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            # batch norm
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0))
            outputs = self.model(inputs)
            # output
            logits, feats = outputs['logits'], outputs['feat']
            # labeled
            logits_x_lb, feats_x_lb = logits[:num_lb], feats[:num_lb]
            # unlabeled
            logits_x_ulb_w, logits_x_ulb_s_0 = logits[num_lb:].chunk(2)
            feats_x_ulb_w, feats_x_ulb_s_0 = feats[num_lb:].chunk(2)

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            with torch.no_grad():
                feats_x_lb = feats_x_lb.detach()
                feats_x_ulb_w = feats_x_ulb_w.detach()
                logits_x_ulb_w = logits_x_ulb_w.detach()

            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s_0}
            
            with torch.no_grad():
                # probs = torch.softmax(logits_x_ulb_w, dim=1)            
                probs_x_ulb_w = self.compute_prob(logits_x_ulb_w)
                # distribution alignment
                probs_x_ulb_w_da = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

                lbs_u_guess_orig, max_probs_orig, mask_orig = self.get_lbs_and_masks(probs_x_ulb_w)
                lbs_u_guess_da, max_probs_da, mask_da = self.get_lbs_and_masks(probs_x_ulb_w_da)

                # label matching
                if self.epoch > 0: 
                    sim = torch.exp(torch.mm(feats_x_ulb_w, self.queue_feats.t()) / self.T)
                    sim = sim * self.queue_conf_set
                    sim = sim / sim.sum(1, keepdim=True)
                    
                    cluster_x_ulb_w = torch.mm(sim, self.queue_probs)
                    cluster_x_ulb_w_ca = self.call_hook("dist_align", "ClusterAlignHook", probs_x_ulb=cluster_x_ulb_w.detach())

                    q_u = self.alpha * probs_x_ulb_w_da + (1 - self.alpha) * cluster_x_ulb_w_ca
                    q_u = q_u.detach()

                    # classification & clustering label  
                    p_max, p_label = torch.max(probs_x_ulb_w_da, dim=1)
                    c_max, c_label = torch.max(cluster_x_ulb_w_ca, dim=1)
                    mask_label_consistency = (p_label == c_label).float()  
                    mask_label_consistency = mask_label_consistency.detach()
                else:
                    q_u = probs_x_ulb_w_da
                    mask_label_consistency = torch.tensor(0.0).to(x_lb.device)

                lbs_u_guess, max_probs, mask = self.get_lbs_and_masks(q_u)

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=q_u, softmax_x_ulb=False)
            unsup_loss = self.consistency_loss(logits_x_ulb_s_0,
                                               q_u,
                                               'ce',
                                               mask=mask)

            # compute contrast loss 
            # supervised type Moco
            # refer to https://github.com/zhangyifei01/MoCo-v2-SupContrast
            if self.epoch > 0:  
                # contrast weight
                with torch.no_grad():
                    # ccssl type weight matrix
                    q_j, l_j = torch.max(self.queue_probs, dim=-1)
                    score_mask = torch.outer(max_probs, q_j)
                    #  target between query and queue
                    contrast_mask = lbs_u_guess.reshape(-1, 1).eq(l_j.reshape(1, -1))
                    # pivot-guided mask, positives selection & data filtering
                    # row_mask = q_i.ge(0).float()
                    row_mask = mask_label_consistency
                    column_mask = self.queue_conf_set
                    select_matrix = torch.outer(row_mask, column_mask)                    
                    # final target
                    target_queue = contrast_mask * score_mask * select_matrix
                    target_key = torch.ones(len(row_mask)).to(probs_x_ulb_w.device).unsqueeze(-1)
                    target = torch.cat([target_key, target_queue], dim=1)
                    target = target / target.sum(1, keepdim=True)
                    target = target.detach()
                    
                # query, key, queue in Moco
                queries = feats_x_ulb_s_0         
                keys = feats_x_ulb_w       
                queue = self.queue_feats.clone().detach()
                # contrast logit
                logits_key = torch.einsum("nc,nc->n", [queries, keys]).unsqueeze(-1)
                logits_queue = torch.einsum('nc,kc->nk', [queries, queue])
                logits = torch.cat([logits_key, logits_queue], dim=1)
                logits /= self.T

                # contrast loss
                loss_sup = (-torch.log_softmax(logits, dim=1) * target).sum(dim=-1, keepdim=True).div(target.sum(dim=-1, keepdim=True) + 1e-5)
                contrast_loss = loss_sup.mean()
            else:
                contrast_loss = torch.tensor(0.0).to(x_lb.device)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_c * contrast_loss

            with torch.no_grad():
                onehot = F.one_hot(y_lb, num_classes=self.num_classes)
                one = torch.ones(len(y_lb)).float().to(x_lb.device)

                feats = torch.cat([feats_x_ulb_w, feats_x_lb], dim=0)
                probs = torch.cat([probs_x_ulb_w, onehot], dim=0)
                pivots = torch.cat([mask, one], dim=0)

                self.update_queue(feats, probs, pivots)

        # calculate pseudo label acc
        right_labels_orig = (lbs_u_guess_orig == y_ulb).float() * mask_orig
        pseudo_label_acc_orig = right_labels_orig.sum() / max(mask_orig.sum(), 1.0)

        # calculate pseudo label acc
        right_labels = (lbs_u_guess == y_ulb).float() * mask
        pseudo_label_acc = right_labels.sum() / max(mask.sum(), 1.0)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         contrast_loss=contrast_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         mask_prob=mask.float().mean().item(),
                                         pseudo_acc=pseudo_label_acc.item(),
                                         mask_prob_orig=mask_orig.float().mean().item(),
                                         pseudo_acc_orig=pseudo_label_acc_orig.item(),
                                         label_consistency=mask_label_consistency.float().mean().item())
        return out_dict, log_dict
    
    def get_save_dict(self):
        save_dict =  super().get_save_dict()
        save_dict['queue_feats'] = self.queue_feats.cpu()
        save_dict['queue_probs'] = self.queue_probs.cpu()
        save_dict['queue_conf_set'] = self.queue_conf_set.cpu()
        save_dict['queue_size'] = self.queue_size
        save_dict['queue_ptr'] = self.queue_ptr
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu() 
        save_dict['c_model'] = None if self.hooks_dict['ClusterAlignHook'].p_model is None else self.hooks_dict['ClusterAlignHook'].p_model.cpu() 
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.queue_feats = checkpoint['queue_feats'].cuda(self.gpu)
        self.queue_probs = checkpoint['queue_probs'].cuda(self.gpu)
        self.queue_conf_set = checkpoint['queue_conf_set'].cuda(self.gpu)
        self.queue_size = checkpoint['queue_size']
        self.queue_ptr = checkpoint['queue_ptr']
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['ClusterAlignHook'].p_model = checkpoint['c_model'].cuda(self.args.gpu)
        return checkpoint


