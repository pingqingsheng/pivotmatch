# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import DistAlignQueueHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, concat_all_gather
from .utils import ot_sinkhorn, random_inital, topk_inital

# pivot contrast for conf + pivot contrast for non-conf

# pivotmatch18 + pivot propagation
# pivot contrast for conf, pivot contrast for non-conf, pivot propagation

class Prototypes(nn.Module):
    def __init__(self, rep_dim, num_prototypes):
        super(Prototypes, self).__init__()
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Linear(rep_dim, num_prototypes, bias=False)

    def update_wegihts(self, q_initial):
        assert self.prototypes.out_features == q_initial.shape[0]
        self.prototypes.weight.data = q_initial.float() 

    def normalize(self):
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.weight.data = w

    def get_values(self, normalize=False):
        w = self.prototypes.weight.data.clone()
        if normalize:
            with torch.no_grad():
                w = F.normalize(w, dim=1)
        return w
    
    def forward(self, z):
        q = self.prototypes(z)
        return q

class PivotMatch_Net(nn.Module):
    def __init__(self, base, proj_size=128, num_pivots=200):
        super(PivotMatch_Net, self).__init__()
        self.backbone = base
        self.num_features = base.num_features
        
        self.mlp_proj = nn.Sequential(*[
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(inplace=False),
            nn.Linear(self.num_features, proj_size)
        ])

        self.num_pivots = num_pivots
        self.pivots_layer = Prototypes(proj_size, num_pivots)
        self.pivots_label = None   

    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out
    
    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        feat_proj = self.l2norm(self.mlp_proj(feat))
        affinity = self.pivots_layer(feat_proj)
        return {'logits':logits, 'feat':feat_proj, 'affinity':affinity}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher

@ALGORITHMS.register('pivotmatch18')
class PivotMatch18(AlgorithmBase):
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
        self.init(T=args.T, p_cutoff=args.p_cutoff, proj_size=args.proj_size, 
                  pivot_p_cutoff=args.pivot_p_cutoff, hard_label=args.hard_label, 
                  queue_batch=args.queue_batch, alpha=args.alpha, da_len=args.da_len)
    
    def init(self, T, p_cutoff, proj_size, pivot_p_cutoff, hard_label, queue_batch, alpha, da_len=0):
        self.T = T 
        self.p_cutoff = p_cutoff
        self.pivot_p_cutoff = pivot_p_cutoff
        self.hard_label = hard_label 
        self.proj_size = proj_size 
        self.queue_batch = queue_batch
        self.alpha = alpha
        self.epoch_warm_up = self.cfg.epoch_warm_up
        # ot para
        self.epsilon1 = self.cfg.epsilon1
        self.epsilon2 = self.cfg.epsilon2

        # TODO：move this part into a hook
        # memory bank
        self.queue_size = int(queue_batch * (self.args.uratio + 1) * self.args.batch_size)
        self.queue_feats_w = torch.zeros(self.queue_size, self.args.proj_size).cuda(self.gpu)
        self.queue_probs = torch.zeros(self.queue_size, self.args.num_classes).cuda(self.gpu)
        self.queue_pivots = torch.zeros(self.queue_size).cuda(self.gpu)
        self.queue_ptr = 0

    def set_hooks(self):
        self.register_hook(DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),  "DistAlignHook")
        self.register_hook(DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),  "ClusterAlignHook")
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
    def update_queue(self, feats_w, probs, pivots):
        if self.distributed and self.world_size > 1:
            feats_w = concat_all_gather(feats_w)
            probs = concat_all_gather(probs)
            pivots = concat_all_gather(pivots)
        # update memory bank
        length = feats_w.shape[0]
        self.queue_feats_w[self.queue_ptr:self.queue_ptr + length, :] = feats_w
        self.queue_probs[self.queue_ptr:self.queue_ptr + length, :] = probs    
        self.queue_pivots[self.queue_ptr:self.queue_ptr + length] = pivots    
        self.queue_ptr = (self.queue_ptr + length) % self.queue_size
        
    @torch.no_grad()
    def get_lbs_and_masks(self, probs):
        max_probs, lbs_u_guess = torch.max(probs, dim=1)
        mask = max_probs.ge(self.p_cutoff).float()
        return lbs_u_guess, max_probs, mask

    def distribution_alignment_ot(self, probs):
        num_data = probs.shape[0]
        probs_all = torch.cat([probs, self.queue_probs], 0)
        probs_ot = ot_sinkhorn(-probs_all, epsilon=self.epsilon2, exp=False)[:num_data]
        probs_ot = probs_ot.detach()
        return probs_ot

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s_0, x_ulb_s_1, y_ulb):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            # batch norm
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s_0))
            outputs = self.model(inputs)
            # output
            logits, feats, affinity = outputs['logits'], outputs['feat'], outputs['affinity']
            # labeled
            logits_x, feats_x = logits[:num_lb], feats[:num_lb]
            # unlabeled
            logits_u_w, logits_u_s0 = logits[num_lb:].chunk(2)
            feats_u_w, feats_u_s0 = feats[num_lb:].chunk(2)
            _, affinity_u_s0 = affinity[num_lb:].chunk(2)
            feat_dict = {'x_lb': feats_x, 'x_ulb_w': feats_u_w, 'x_ulb_s': feats_u_s0}
            
            with torch.no_grad():
                # no grad feature
                feats_x = feats_x.detach() 
                logits_u_w, feats_u_w = logits_u_w.detach(), feats_u_w.detach()
                # distribution alignment
                probs = torch.softmax(logits_u_w, dim=1)   
                probs_u_w_da = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs.detach())
                probs_u_w_ot = self.distribution_alignment_ot(probs_u_w)
                # get labels and mask (test)
                lbs_u_guess_orig, max_probs_orig, mask_orig = self.get_lbs_and_masks(probs_u_w)
                lbs_u_guess_da, max_probs_da, mask_da = self.get_lbs_and_masks(probs_u_w_da)
                lbs_u_guess_ot, max_probs_ot, mask_ot = self.get_lbs_and_masks(probs_u_w_ot)

                # label matching
                if self.epoch > 0: 
                    sim = torch.exp(torch.mm(feats_u_w, self.queue_feats.t()) / self.T)
                    sim = sim * self.queue_pivots
                    sim = sim / sim.sum(1, keepdim=True)
                    
                    cluster_u_w = torch.mm(sim, self.queue_probs)
                    cluster_u_w_ca = self.call_hook("dist_align", "ClusterAlignHook", probs_x_ulb=cluster_u_w.detach())

                    q_u_final = self.alpha * probs_u_w_da + (1 - self.alpha) * cluster_u_w_ca
                    q_u_final2 = self.alpha * probs_u_w_ot + (1 - self.alpha) * cluster_u_w_ca
                    q_u_final = q_u_final.detach()
                    q_u_final2 = q_u_final2.detach()

                    # classification & clustering label  
                    p_max, p_label = torch.max(probs_u_w_da, dim=1)
                    c_max, c_label = torch.max(cluster_u_w_ca, dim=1)
                    mask_label_consistency = (p_label == c_label).float().detach()  
                else:
                    q_u_final = probs_u_w_da.clone()
                    q_u_final2 = probs_u_w_ot.clone()
                    mask_label_consistency = torch.tensor(0.0).to(probs_u_w_da.device)

                # get labels and mask
                lbs_u_guess, max_probs, mask = self.get_lbs_and_masks(q_u_final)
                lbs_u_guess2, max_probs2, mask2 = self.get_lbs_and_masks(q_u_final2)

            # compute mask
            unsup_loss = self.consistency_loss(logits_u_s0, q_u_final, 'ce', mask=mask)

            # compute contrast loss 
            # supervised type Moco
            # refer to https://github.com/zhangyifei01/MoCo-v2-SupContrast
            if self.epoch > 0:  
                # query, key, queue in Moco
                queries = feats_u_s0                
                queue_feat_w = self.queue_feats_w.clone().detach()
                queue_probs = self.queue_probs.clone().detach()

                # contrast logit
                logits = torch.einsum('nc,kc->nk', [queries, queue_feat_w])
                logits /= self.T

                # contrast weight
                
                with torch.no_grad():
                    # ccssl type weight matrix
                    q_i, l_i = torch.max(q_u_final, dim=-1)
                    q_j, l_j = torch.max(queue_probs, dim=-1)
                    contrast_mask =  torch.mm(F.one_hot(l_i, self.num_classes).float(), F.one_hot(l_j, self.num_classes).float().t())
                    score_mask = torch.outer(q_i, q_j)
                    # pivot-guided mask, positives selection & data filtering
                    row_mask = q_i.ge(0).float()
                    column_mask = self.queue_pivots
                    select_matrix = torch.outer(row_mask, column_mask)
                    # contrast target
                    # supervised target between query and queue
                    target_queue = contrast_mask * score_mask * select_matrix
                    # row_mask = mask_label_consistency
                    # column_mask = self.queue_pivots
                    # select_matrix = torch.outer(row_mask, column_mask)
                    # score_mask = torch.mm(q_u, queue_probs.t())
                    # target_queue = score_mask * select_matrix
                    
                    # final target
                    target = target_queue
                    target = target / target.sum(1, keepdim=True)
                    target.detach()
                # contrast loss
                loss_sup = (-torch.log_softmax(logits, dim=1) * target).sum(dim=-1, keepdim=True).div(target.sum(dim=-1, keepdim=True) + 1e-5)
                contrast_loss = (loss_sup * mask_label_consistency).mean()
            else:
                contrast_loss = torch.tensor(0.0).to(x_lb.device)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_c * contrast_loss

            with torch.no_grad():
                onehot = F.one_hot(y_lb, num_classes=self.num_classes)
                one = torch.ones(len(y_lb)).float().to(x_lb.device)

                feats_w = torch.cat([feats_x_ulb_w, feats_x_lb], dim=0)
                probs = torch.cat([probs_x_ulb_w, onehot], dim=0)
                pivots = torch.cat([mask_pivot, one], dim=0)

                self.update_queue(feats_w, probs, pivots)

        # calculate pseudo label acc
        right_labels_orig = (lbs_u_guess_orig == y_ulb).float() * mask_orig
        pseudo_label_acc_orig = right_labels_orig.sum() / max(mask_orig.sum(), 1.0)

        # calculate pseudo label acc
        right_labels = (lbs_u_guess == y_ulb).float() * mask
        pseudo_label_acc = right_labels.sum() / max(mask.sum(), 1.0)

        # calculate pseudo label acc
        right_labels = (lbs_u_guess == y_ulb).float() * mask_pivot
        pseudo_label_acc_pivot = right_labels.sum() / max(mask_pivot.sum(), 1.0)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(
            loss=total_loss.item(), 
            loss_x=sup_loss.item(), 
            loss_u=unsup_loss=unsup_loss.item(), 
            loss_c=contrast_loss.item(), 
            mask_prob=mask.float().mean().item(),
            pseudo_acc=pseudo_label_acc.item(),
            mask_prob_pivot=mask_pivot.float().mean().item(),
            pseudo_acc_pivot=pseudo_label_acc_pivot.item(),
            mask_prob_orig=mask_orig.float().mean().item(),
            pseudo_acc_orig=pseudo_label_acc_orig.item(),
            label_consistency=mask_label_consistency.float().mean().item())
        return out_dict, log_dict
    
    def get_save_dict(self):
        save_dict =  super().get_save_dict()
        save_dict['queue_feats_w'] = self.queue_feats_w.cpu()
        # save_dict['queue_feats_s'] = self.queue_feats_s.cpu()
        save_dict['queue_probs'] = self.queue_probs.cpu()
        save_dict['queue_pivots'] = self.queue_pivots.cpu()
        save_dict['queue_size'] = self.queue_size
        save_dict['queue_ptr'] = self.queue_ptr
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu() 
        save_dict['p_model_ptr'] = self.hooks_dict['DistAlignHook'].p_model_ptr.cpu()
        save_dict['c_model'] = self.hooks_dict['ClusterAlignHook'].p_model.cpu() 
        save_dict['c_model_ptr'] = self.hooks_dict['ClusterAlignHook'].p_model_ptr.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.queue_feats_w = checkpoint['queue_feats_w'].cuda(self.gpu)
        self.queue_feats_s = checkpoint['queue_feats_s'].cuda(self.gpu)
        self.queue_probs = checkpoint['queue_probs'].cuda(self.gpu)
        self.queue_pivots = checkpoint['queue_pivots'].cuda(self.gpu)
        self.queue_size = checkpoint['queue_size']
        self.queue_ptr = checkpoint['queue_ptr']
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_model_ptr = checkpoint['p_model_ptr'].cuda(self.args.gpu)
        self.hooks_dict['ClusterAlignHook'].p_model = checkpoint['c_model'].cuda(self.args.gpu)
        self.hooks_dict['ClusterAlignHook'].p_model_ptr = checkpoint['c_model_ptr'].cuda(self.args.gpu)
        # self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        # self.hooks_dict['DistAlignHook'].p_target_ptr = checkpoint['p_target_ptr'].cuda(self.args.gpu)
        return checkpoint


    @staticmethod
    def get_argument():
        return [
            # SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--pivot_p_cutoff', float, 0.95),
            SSL_Argument('--contrast_loss_ratio', float, 1.0),
            SSL_Argument('--proj_size', int, 128),
            SSL_Argument('--queue_batch', int, 128),
            SSL_Argument('--alpha', float, 0.9),
            SSL_Argument('--da_len', int, 256),
        ]

