##adapted from https://github.com/tkipf/gae and https://github.com/zfjsail/gae-pytorch/blob/master/gae/optimizer.py ##

import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np


def optimizerAE(preds, labels, mu, logvar, num_nodes, pos_weight, norm):
    cost=norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight,reduction='mean')
    return cost

def optimizerVAE(preds, labels, mu, logvar, num_nodes, pos_weight, norm):
    cost=norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight,reduction='mean')
    kl= (0.5 / num_nodes) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    cost-=kl
    return cost


def optimizer_kl(mu, logvar, nodemask=None,reduction='mean'):
    if reduction=='mean':
        f=torch.mean
        if nodemask is None:
            s=mu.size()[0]
        else:
            s=nodemask.size()[0]
    elif reduction=='sum':
        f=torch.sum
        s=1
    if nodemask is None:
        kl= -(0.5 / s) * f(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return kl
    kl= -(0.5 / s) * f(torch.sum(1 + 2 * logvar[nodemask] - mu[nodemask].pow(2) - logvar[nodemask].exp().pow(2), 1))
    return kl

def optimizer_CE(preds, labels, pos_weight, norm,nodemask=None):
    if nodemask is None:
        cost=norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight,reduction='mean')
        return cost
    cost=norm * F.binary_cross_entropy_with_logits(preds[nodemask,:][:,nodemask], labels[nodemask,:][:,nodemask], pos_weight=pos_weight,reduction='mean')
    return cost

def optimizer_CEclf(preds, labels, nodemask):
    cost= F.binary_cross_entropy_with_logits(preds[nodemask,:], labels[nodemask,:])
    return cost

def optimizer_MSE(preds, inputs,mask,reconWeight,mse):
    cost = mse(preds[mask], inputs[mask])*reconWeight
    return cost
               
def optimizer_nb(preds,y_true,mask,reconWeight,eps = 1e-10,ifmean=True):
    #adapted from https://github.com/theislab/dca/blob/master/dca/loss.py
    output,pi,theta,y_pred=preds
    nbloss1=torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
    nbloss2=(theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))
    nbloss=nbloss1+nbloss2
    
#     nbloss=torch.where(torch.isnan(nbloss), torch.zeros_like(nbloss)+np.inf, nbloss)
    if ifmean:
        return torch.mean(nbloss[mask])*reconWeight
    else:
        return nbloss
    
def optimizer_zinb(preds,y_true,mask,reconWeight,ridgePi,y_true_raw,eps = 1e-10):
    output,pi,theta,y_pred=preds
    nb_case=optimizer_nb(preds,y_true,mask,reconWeight,eps = 1e-10,ifmean=False)- torch.log(pi+eps)
    
    zero_nb = torch.pow(theta/(theta+y_pred+eps), theta)
    zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
    result = torch.where(torch.lt(y_true_raw, 1), zero_case, nb_case)
    ridge = ridgePi*pi*pi
    result += ridge
    result=torch.mean(result[mask])
    
#     result=torch.where(torch.isnan(result), torch.zeros_like(result)+np.inf, result)
#     print(result.shape)
    return result*reconWeight
    
    
def accuracy(output, labels):
    preds = (torch.sigmoid(output)>0.5).double()
    correct = preds.eq(labels.double()).double()
    return torch.mean(correct)

def get_roc_score(edges_pos, edges_neg, adj_rec,adj_orig):
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(torch.sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(torch.sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

# class OptimizerAE(object):
#     def __init__(self, preds, labels, pos_weight, norm):
#         preds_sub = preds
#         labels_sub = labels

#         self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
#         self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

#         self.opt_op = self.optimizer.minimize(self.cost)
#         self.grads_vars = self.optimizer.compute_gradients(self.cost)

#         self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
#                                            tf.cast(labels_sub, tf.int32))
#         self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

# class OptimizerVAE(object):
#     def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
#         preds_sub = preds
#         labels_sub = labels

#         self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
#         self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

#         # Latent loss
#         self.log_lik = self.cost
#         self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
#                                                                    tf.square(tf.exp(model.z_log_std)), 1))
#         self.cost -= self.kl

#         self.opt_op = self.optimizer.minimize(self.cost)
#         self.grads_vars = self.optimizer.compute_gradients(self.cost)

#         self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
#                                            tf.cast(labels_sub, tf.int32))
#         self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
