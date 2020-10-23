##adapted from https://github.com/tkipf/gae and https://github.com/zfjsail/gae-pytorch/blob/master/gae/optimizer.py ##

import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def optimizerAE(preds, labels, mu, num_nodes, pos_weight, norm):
    cost=norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight,reduction='mean')
    return cost

def optimizerVAE(preds, labels, mu, logvar, num_nodes, pos_weight, norm):
    cost=norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight,reduction='mean')
    kl= (0.5 / num_nodes) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    cost-=kl
    return cost

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
