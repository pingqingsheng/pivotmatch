# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import DistAlignQueueHook, DistAlignEMAHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, concat_all_gather
from .utils import ot_sinkhorn, random_inital, topk_inital
from collections import Counter
import numpy as np

# weak + hard for pivot, pivot + queue contrast representation, pivot propagation
#  

class Prototypes(nn.Module):
    def __init__(self, rep_dim, num_prototypes):
        super(Prototypes, self).__init__()
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Linear(rep_dim, num_prototypes, bias=False)

    def update_wegihts(self, q):
        assert self.prototypes.out_features == q.shape[0]
        self.prototypes.weight.data = q.float() 

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
    def __init__(self, base, proj_size=128, num_pivots=50):
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
    
    def forward(self, x, pivots_grad_only=False, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        feat_proj = self.l2norm(self.mlp_proj(feat))
        if pivots_grad_only:
            affinity = self.pivots_layer(feat_proj.clone().detach())
        else:
            affinity = self.pivots_layer(feat_proj)
        return {'logits':logits, 'feat':feat_proj, 'affinity':affinity}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher

@ALGORITHMS.register('pivotmatch46')
class PivotMatch46(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # pivotmatch specified arguments
        self.args = args
        self.use_ema_teacher = True
        if args.dataset in ['cifar10', 'cifar100', 'svhn', 'superks', 'tissuemnist', 'eurosat', 'superbks', 'esc50', 'gtzan', 'urbansound8k', 'aclImdb', 'ag_news', 'dbpedia']:
            self.use_ema_teacher = False

        self.init_params()
    
    def init_params(self):
        # base para
        self.temperature = self.args.temperature 
        self.threshold = self.args.threshold
        self.hard_label = self.args.hard_label 
        self.num_classes = self.args.num_classes
        self.lambda_u = self.args.lambda_u
        self.lambda_p = self.args.lambda_p
        self.lambda_c = self.args.lambda_c
        self.alpha = self.args.alpha
        self.queue_batch = self.args.queue_batch 
        self.epoch_warm_up = self.args.epoch_warm_up
        self.num_pivots = self.args.num_pivots
        # ot para
        self.epsilon1 = self.args.epsilon1
        self.epsilon2 = self.args.epsilon2
        # loss function
        self.loss_p = nn.CrossEntropyLoss()
        # self.loss_c = nn.CrossEntropyLoss(reduction='none')
        self.loss_c = nn.CrossEntropyLoss()
        # memory bank
        # para
        self.queue_ptr = 0
        self.queue_size = int(self.queue_batch * (self.args.uratio + 1) * self.args.batch_size)
        # queue
        self.queue_feats = torch.zeros(self.queue_size, self.args.proj_size).cuda(self.gpu)
        self.queue_probs = torch.zeros(self.queue_size, self.args.num_classes).cuda(self.gpu)
        self.queue_conf_set = torch.zeros(self.queue_size).cuda(self.gpu)

    def set_hooks(self):     
        self.register_hook(DistAlignEMAHook(num_classes=self.num_classes, p_target_type='uniform'),  "DistAlignHook")
        # self.register_hook(DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),  "DistAlignHook")
        # self.register_hook(DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),  "ClusterAlignHook")
        # self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self): 
        model = super().set_model()
        model = PivotMatch_Net(model, proj_size=self.args.proj_size, num_pivots=self.args.num_pivots)
        return model
    
    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = PivotMatch_Net(ema_model, proj_size=self.args.proj_size, num_pivots=self.args.num_pivots)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model    

    @torch.no_grad()
    def update_queue(self, feats_x, targets_x, feats_u_w, probs_u_w, mask, total_batch_size):
        if self.distributed and self.world_size > 1:
            feats_x = concat_all_gather(feats_x)
            targets_x = concat_all_gather(targets_x)
            pivots = concat_all_gather(pivots)
            feats_u_w = concat_all_gather(feats_u_w)
            probs_u_w = concat_all_gather(probs_u_w)
            mask = concat_all_gather(mask)
        # info
        onehot = F.one_hot(targets_x, num_classes=self.args.num_classes)
        one = torch.ones(len(targets_x)).float().to(mask.device)
        # stored info
        feats_w = torch.cat([feats_u_w, feats_x], dim=0)
        probs_w = torch.cat([probs_u_w, onehot], dim=0)
        conf_set = torch.cat([mask, one], dim=0)
        # insert
        # n = batch_size + batch_size_u
        n = total_batch_size
        self.queue_feats[self.queue_ptr:self.queue_ptr + n, :] = feats_w
        self.queue_probs[self.queue_ptr:self.queue_ptr + n, :] = probs_w
        self.queue_conf_set[self.queue_ptr:self.queue_ptr + n] = conf_set
        self.queue_ptr = (self.queue_ptr + n) % self.queue_size
        
    @torch.no_grad()
    def init_pivots(self, method='topk', threshold=0.95):
        idx = torch.where(self.queue_conf_set > 0)[0]
        x_conf, x_target = torch.max(self.queue_probs, dim=-1)
        # pivot candidate
        x_feature = self.queue_feats[idx]
        x_target = x_target[idx]
        x_conf = x_conf[idx]
        # selection
        if method == 'random':
            pivots_initial, pivots_label = random_inital(x_feature, x_target, x_conf, self.num_pivots, self.num_classes, threshold)
        elif method == 'topk':
            pivots_initial, pivots_label = topk_inital(x_feature, x_target, x_conf, self.num_pivots, self.num_classes)
        # initial
        self.model.pivots_layer.update_wegihts(pivots_initial)
        self.ema_model.pivots_layer.update_wegihts(pivots_initial)
        self.model.pivots_label = pivots_label
        self.ema_model.pivots_label = pivots_label
        return pivots_initial, pivots_label

    @torch.no_grad()
    def get_lbs_and_masks(self, probs):
        max_probs, lbs_u_guess = torch.max(probs, dim=1)
        mask = max_probs.ge(self.threshold).float()
        return lbs_u_guess, max_probs, mask

    @torch.no_grad()
    def get_clustering_label(self, pivots, pivots_label, feats):
        num_feats = feats.shape[0]
        # aggregated prob
        feats_all = torch.cat([feats, self.queue_feats], 0)
        sim = torch.mm(feats_all, pivots.t())
        weight = torch.exp(sim / self.temperature)
        weight = weight / weight.sum(1, keepdim=True)
        prob_agg = torch.mm(weight, F.one_hot(pivots_label, self.num_classes).float())
        # clustering label
        cluster_ot = ot_sinkhorn(-prob_agg, epsilon=self.epsilon2, exp=False)[:num_feats]
        cluster_ot /= cluster_ot.sum(1, keepdim=True) 
        cluster_ot = cluster_ot.detach()
        return cluster_ot

    @torch.no_grad()
    def get_pivot_code(self, pivots, pivots_label, feats, pseudo_label, idx_conf):
        num_pivots = pivots.shape[0]
        # confident samples
        num_conf = idx_conf.shape[0]
        idx_conf_queue = torch.where(self.queue_conf_set > 0)[0]
        # affinity
        feats_batch = feats[idx_conf]
        feats_queue = self.queue_feats[idx_conf_queue]
        feats_sample = torch.cat([feats_batch, feats_queue], 0) 
        affinity_sample = torch.mm(feats_sample, pivots.t())
        # pseudo label
        pseudo_label_batch = pseudo_label[idx_conf]
        pseudo_label_queue = torch.max(self.queue_probs, dim=-1)[1]
        pseudo_label_queue = pseudo_label_queue[idx_conf_queue]
        pseudo_label_sample = torch.cat([pseudo_label_batch, pseudo_label_queue], 0)
        # class-aware affinity
        label_mask = pseudo_label_sample.reshape(-1, 1).eq(pivots_label.reshape(1, -1))
        affinity_sample = affinity_sample * label_mask.float()
        # marginal distribution
        ## column 
        c = torch.ones(num_pivots).to(feats.device) / num_pivots
        ## row 
        count = Counter(pseudo_label_sample.cpu().numpy())
        count = [count[i] for i in range(self.num_classes)]
        prob = np.array([1 / (self.num_classes * i) for i in count])
        r = prob[pseudo_label_sample.cpu().numpy()]
        r = torch.from_numpy(r).to(feats.device)
        # OT clustering
        q_sample = ot_sinkhorn(-affinity_sample, r, c, epsilon=self.epsilon1)
        q_sample *= label_mask.float()
        q_sample /= q_sample.sum(1, keepdim=True) 
        pivot_code = q_sample[:num_conf]
        return pivot_code.detach(), q_sample.detach(), feats_sample.detach()

    def get_contrast_label(self, pivots_label, pseudo_label, pivot_code, idx_non_conf):
        # contrast target
        # target between query and key
        target_pivot_non_conf = pseudo_label[idx_non_conf].reshape(-1, 1).eq(pivots_label.reshape(1, -1))
        target_pivot = torch.cat([pivot_code, target_pivot_non_conf], dim=0)
        #  target between query and queue
        target_queue = torch.zeros(pseudo_label.shape[0], self.queue_feats.shape[0]).to(pseudo_label.device)
        # final target
        target = torch.cat([target_pivot, target_queue], dim=1)
        target = target / target.sum(1, keepdim=True)
        target = target.detach()
        # logit mask
        column_mask = self.queue_conf_set
        row_mask = torch.ones(pseudo_label.shape[0]).to(pseudo_label.device)
        select_matrix = torch.outer(row_mask, column_mask)
        pseudo_label_queue = torch.max(self.queue_probs, dim=-1)[1]
        label_matrix = pseudo_label.reshape(-1, 1).eq(pseudo_label_queue.reshape(1, -1))
        queue_mask = 1 - select_matrix * label_matrix
        pivots_mask = torch.ones(pseudo_label.shape[0], pivots_label.shape[0]).to(pseudo_label.device)
        logits_mask = torch.cat([pivots_mask, queue_mask], dim=1)
        logits_mask = logits_mask.detach()
        return target, logits_mask

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, y_ulb):
        batch_size_x = y_lb.shape[0]
        batch_size_u = y_ulb.shape[0]
        targets_x = y_lb
        targets_u = y_ulb

        self.model.pivots_layer.normalize()
        self.ema_model.pivots_layer.normalize()

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            # batch norm
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s), dim=0)
            outputs = self.model(inputs, pivots_grad_only=True)
            # output
            logits, feats, affinity = outputs['logits'], outputs['feat'], outputs['affinity']
            # labeled
            logits_x, feats_x = logits[:batch_size_x], feats[:batch_size_x]
            # unlabeled
            logits_u_w, logits_u_s = logits[batch_size_x:].chunk(2)
            feats_u_w, feats_u_s = feats[batch_size_x:].chunk(2)
            affinity_u_w, affinity_u_s = affinity[batch_size_x:].chunk(2)
            feat_dict = {'x_lb': feats_x, 'x_ulb_w': feats_u_w, 'x_ulb_s': feats_u_s}
            
            with torch.no_grad():
                # no grad feature
                feats_x = feats_x.detach() 
                logits_u_w, feats_u_w = logits_u_w.detach(), feats_u_w.detach()
                # distribution alignment
                probs_u_w = torch.softmax(logits_u_w, dim=1)   
                probs_u_w_da = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_u_w.detach())
                # probs_u_w_ot = self.distribution_alignment_ot(probs_u_w)
                # get labels and mask (test)
                lbs_u_guess_orig, max_probs_orig, mask_orig = self.get_lbs_and_masks(probs_u_w)
                lbs_u_guess_da, max_probs_da, mask_da = self.get_lbs_and_masks(probs_u_w_da)

                # initiate pivots 
                if self.epoch >= self.epoch_warm_up:
                    if self.epoch == self.epoch_warm_up:
                        pivots, pivots_label = self.init_pivots()
                    else:
                        pivots = self.model.pivots_layer.get_values().detach()
                        pivots_label = self.model.pivots_label

                # label propagation with pivots 
                if self.epoch >= self.epoch_warm_up:
                    # update labels
                    cluster_u_w_ot = self.get_clustering_label(pivots, pivots_label, feats_u_w)
                    q_u_final = self.alpha * probs_u_w_da + (1 - self.alpha) * cluster_u_w_ot
                    q_u_final = q_u_final.detach()
                    # classification & clustering label  
                    p_max, p_label = torch.max(probs_u_w_da, dim=1)
                    c_max, c_label = torch.max(cluster_u_w_ot, dim=1)
                    mask_label_consistency = (p_label == c_label).float().detach()  
                else:
                    cluster_u_w_ot = torch.rand(probs_u_w_da.shape).to(probs_u_w_da.device)
                    q_u_final = probs_u_w_da.clone()
                    mask_label_consistency = torch.tensor(0.0).to(probs_u_w_da.device)
                # get labels and mask
                lbs_u_guess, max_probs, mask = self.get_lbs_and_masks(q_u_final)
                lbs_u_guess_cluster, max_probs_cluster, mask_cluster = self.get_lbs_and_masks(cluster_u_w_ot)
            
                # target for pivots learning & representation learning 
                if self.epoch >= self.epoch_warm_up:
                    # get code for pivots learning
                    idx_conf = torch.where(mask > 0)[0]
                    idx_non_conf = torch.where(mask < 0.5)[0]
                    pivot_code, q_sample, feats_sample = self.get_pivot_code(pivots, pivots_label, feats_u_w, lbs_u_guess, idx_conf)
                    # get contrast target for representation learning 
                    contrast_target, logits_mask = self.get_contrast_label(pivots_label, lbs_u_guess, pivot_code, idx_non_conf)
                    # update pivots
                    pivots_new = []
                    q_sample = torch.max(q_sample, dim=1)[1]
                    for class_id in range(pivots.shape[0]):
                        idx_class = torch.where(q_sample == class_id)[0]
                        feature_selected = feats_sample[idx_class]
                        pivots_new.append(feature_selected.mean(0))
                    pivots_new = torch.stack(pivots_new, 0)
                    pivots_new = F.normalize(pivots_new, 1)
                    self.model.pivots_layer.update_wegihts(pivots_new)

            # supervision loss
            loss_x = self.ce_loss(logits_x, targets_x, reduction='mean')

            # unsupervised loss
            loss_u = self.consistency_loss(logits_u_s, q_u_final, 'ce', mask=mask)

            # pivot loss & contrast loss
            loss_p = torch.tensor(0, dtype=torch.float).to(logits_x.device)
            if self.epoch >= self.epoch_warm_up:
                ## contrast loss
                queries = torch.cat([feats_u_s[idx_conf], feats_u_s[idx_non_conf]], dim=0)
                queue = self.queue_feats.clone().detach()
                enqueue = torch.cat([pivots, queue], dim=0)
                # contrast logit
                logits = torch.einsum('nc,kc->nk', [queries, enqueue])
                logits = logits * logits_mask
                # contrast loss
                loss_c = self.loss_c(logits / self.temperature, contrast_target)              
            else:
                loss_c = torch.tensor(0, dtype=torch.float).to(logits_x.device)

            total_loss = loss_x + self.lambda_u * loss_u + self.lambda_p * loss_p + self.lambda_c * loss_c

            with torch.no_grad():
                self.update_queue(feats_x, targets_x, feats_u_w, probs_u_w, mask, batch_size_x + batch_size_u)

        # calculate pseudo label acc
        right_labels = (lbs_u_guess == y_ulb).float() * mask
        pseudo_label_acc = right_labels.sum() / max(mask.sum(), 1.0)

        # calculate pseudo label acc
        right_labels_orig = (lbs_u_guess_orig == targets_u).float() * mask_orig
        pseudo_label_acc_orig = right_labels_orig.sum() / max(mask_orig.sum(), 1.0)

        # calculate pseudo label acc
        right_labels_da = (lbs_u_guess_da == targets_u).float() * mask_da
        pseudo_label_acc_da = right_labels_da.sum() / max(mask_da.sum(), 1.0)

        # calculate pseudo label acc
        right_labels_cluster = (lbs_u_guess_cluster == targets_u).float() * mask_cluster
        pseudo_label_acc_cluster = right_labels_cluster.sum() / max(mask_cluster.sum(), 1.0)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(
            loss=total_loss.item(), 
            loss_x=loss_x.item(), 
            loss_u=loss_u.item(), 
            loss_c=loss_c.item(), 
            loss_p=loss_p.item(), 
            mask_prob=mask.float().mean().item(),
            pseudo_acc=pseudo_label_acc.item(),
            mask_prob_orig=mask_orig.float().mean().item(),
            pseudo_acc_orig=pseudo_label_acc_orig.item(),
            mask_da=mask_da.float().mean().item(),
            pseudo_label_acc_da=pseudo_label_acc_da.item(),
            mask_cluster=mask_cluster.float().mean().item(),
            pseudo_label_acc_cluster=pseudo_label_acc_cluster.item(),
            queue_conf_set=self.queue_conf_set.float().mean().item(),
            mask_label_consistency=mask_label_consistency.float().mean().item())
        return out_dict, log_dict
    
    def get_save_dict(self):
        save_dict =  super().get_save_dict()
        save_dict['queue_feats'] = self.queue_feats.cpu()
        save_dict['queue_probs'] = self.queue_probs.cpu()
        save_dict['queue_conf_set'] = self.queue_conf_set.cpu()
        save_dict['queue_size'] = self.queue_size
        save_dict['queue_ptr'] = self.queue_ptr
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu() 
        # save_dict['p_model_ptr'] = self.hooks_dict['DistAlignHook'].p_model_ptr.cpu() 
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.queue_feats = checkpoint['queue_feats'].cuda(self.gpu)
        self.queue_probs = checkpoint['queue_probs'].cuda(self.gpu)
        self.queue_queue_conf_setivots = checkpoint['queue_conf_set'].cuda(self.gpu)
        self.queue_size = checkpoint['queue_size']
        self.queue_ptr = checkpoint['queue_ptr']
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        # self.hooks_dict['DistAlignHook'].p_model_ptr = checkpoint['p_model_ptr'].cuda(self.args.gpu)
        return checkpoint


