#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Disk Embedding Model Original Version.
"""

from .dag_emb_model import DAGEmbeddingModel, DAGEmbeddingKeyedVectors
from .metric_space import get_metric_space, QuasimetricSpaceBase, SphericalSpace
from .loss_function import get_loss_function
import numpy as np

try:
    from autograd import grad  # Only required for optionally verifying gradients while training
    from autograd import numpy as grad_np
    AUTOGRAD_PRESENT = True
except ImportError:
    AUTOGRAD_PRESENT = False

# Cosine clipping epsilon
EPS = 1e-7

class DiskEmbeddingModel(DAGEmbeddingModel):
    """Class for training, using and evaluating Order Embeddings."""
    def __init__(self,
                 train_data,
                 dim=50,
                 init_range=(-0.1, 0.1),
                 lr=0.01,
                 seed=0,
                 logger=None,
                 opt='disk_rsgd',
                 burn_in=0,

                 num_negative=1,
                 ### How to sample negatives for an edge (u,v)
                 neg_sampl_strategy='all',  # 'all' (all nodes for negative sampling) or 'true_neg' (only nodes not connected)
                 where_not_to_sample='children',  # both or ancestors or children. Has no effect if neg_sampl_strategy = 'all'.
                 neg_edges_attach='both',  # How to form negative edges: 'parent' (u,v') or 'child' (u', v) or 'both'
                 always_v_in_neg=False,
                 neg_sampling_power=0.0,  # 0 for uniform, 1 for unigram, 0.75 for word2vec

                 metric='euc',
                 loss='relu',
                 margin=1.0,  # Margin for the OE loss.
                 radius_range='none',

                 # ### init_vector_conv: how to convert pretrained vector
                 # - poincare, poincare2sphere: PoincareNIPS -> Spherical DE. Uses init_vector_k.
                 # - poincare2lorentz: PoincareNIPS -> Hyperbolic DE.
                 # - centers: n-1 point-wise embedding models -> DE.
                 init_vector_conv='none',
                 init_vector_k=0.1, # used to convert initial vector for spherical model. used only if init_vector_conv='poincare2sphere'
                 init_vector_eps=1e-5, # used to convert Poincare ball to Lorentz model. used only if init_vector_conv='poincare2lorentz'
                 ):

        self.metric = get_metric_space(metric, dim-1)

        self.actual_dim = dim
        self.nomial_dim = self.metric.nomial_dim + 1


        self.loss_function = get_loss_function(loss, margin=margin)
        self.margin = margin
        self.radius_range=radius_range

        assert radius_range in ['none', 'nonnegative', 'child_of_origin', 'parent_of_origin']
        assert opt == 'disk_rsgd'
        assert neg_sampl_strategy == 'all'
        assert neg_edges_attach == 'both'

        assert init_vector_conv in ['none', 'poincare', 'poincare2sphere', 'poincare2lorentz', 'centers']
        self.init_vector_conv = init_vector_conv
        self.init_vector_k = init_vector_k
        self.init_vector_eps = init_vector_eps

        # patch keyed vectors to recieve
        def _keyed_vector_generator():
            return DiskEmbeddingKeyedVectors(self.metric)

        super().__init__(train_data=train_data,
                         dim=self.nomial_dim,
                         init_range=init_range,
                         lr=lr,
                         opt=opt,
                         burn_in=burn_in,
                         seed=seed,
                         logger=logger,
                         # BatchClass=DiskEmbeddingBatch,
                         KeyedVectorsClass=_keyed_vector_generator,
                         num_negative=num_negative,
                         neg_sampl_strategy=neg_sampl_strategy,
                         where_not_to_sample=where_not_to_sample,
                         always_v_in_neg=always_v_in_neg,
                         neg_sampling_power=neg_sampling_power,
                         neg_edges_attach=neg_edges_attach)

    def _clip_vectors(self, vectors, axis=-1):
        radius, centers = np.split(vectors, [1], axis=axis)

        # clip centers
        centers = self.metric.clip_vectors(centers, axis=axis)

        # clip radius
        if self.radius_range == 'nonnegative':
            radius = np.maximum(0, radius)

        elif self.radius_range == 'child_of_origin':
            dist = np.expand_dims(self.metric.compute_distance(np.zeros_like(centers), centers, axis=axis), axis=axis)
            radius = np.minimum(-dist, radius)
        elif self.radius_range == 'parent_of_origin':
            dist = np.expand_dims(self.metric.compute_distance(centers, np.zeros_like(centers), axis=axis), axis=axis)
            radius = np.maximum(dist, radius)

        return np.concatenate((radius, centers), axis=axis)


    def supply_init_vectors(self, vectors):
        vectors = np.asarray(vectors)
        if self.init_vector_conv == 'none':
            self.kv.syn0[:] = vectors

        elif self.init_vector_conv in ['poincare', 'poincare2sphere']:
            if not isinstance(self.metric, SphericalSpace):
                raise ValueError("init_vector_conv = 'poincare' is only available for Spherical Disk Embeddings.")

            K = self.init_vector_k
            t0 = np.arctan(2*K)

            norm = np.linalg.norm(vectors, axis=1)

            radii = np.arcsin(np.clip(0.5*(1+norm**2)/norm*np.sin(t0), 0,1)) - t0
            centers = vectors / norm[:,np.newaxis]

            self.kv.syn0[:,0] = radii
            self.kv.syn0[:,1:] = centers

        elif self.init_vector_conv == 'poincare2lorentz':
            eps = self.init_vector_eps
            norm = np.linalg.norm(vectors, axis=1)

            clipped_vectors = np.where(norm[:,np.newaxis] > 1-eps, vectors*(1-eps)/norm[:,np.newaxis], vectors)
            norm = np.clip(norm, 0, 1-eps)

            self.kv.syn0[:,1] = (1+norm**2) / (1-norm**2)
            self.kv.syn0[:,2:] = clipped_vectors * 2 / (1-norm[:,np.newaxis]**2)

        elif self.init_vector_conv == 'centers':
            self.kv.syn0[:,1:] = vectors

        else:
            raise ValueError("Unknown conversion method")




    def _sample_pairs(self, n):
        max_size = self.kv.syn0.shape[0]

        lefts = self._np_rand.choice(max_size, n)

        rights = self._np_rand.choice(max_size-1, n)
        rights[rights >= lefts] += 1 # avoid identical pairs

        return np.stack((lefts, rights), axis=1)

    def _train_on_batch(self, pos_relations):
        num_pos = len(pos_relations)
        num_neg = int(self.num_negative * num_pos)

        pos_left_indices, pos_right_indices = np.array(pos_relations).T
        neg_left_indices, neg_right_indices = self._sample_pairs(num_neg).T

        left_indices = np.concatenate((pos_left_indices, neg_left_indices))
        right_indices = np.concatenate((pos_right_indices, neg_right_indices))

        labels = np.concatenate((np.ones(num_pos, dtype=bool), np.zeros(num_neg, dtype=bool)))
        left_vectors = self.kv.syn0[left_indices]
        right_vectors = self.kv.syn0[right_indices]

        loss, pos_loss, neg_loss = self.compute_loss(left_vectors, right_vectors, labels, True)
        left_gradients, right_gradients = self.compute_loss_gradients(left_vectors, right_vectors, labels)


        gradients = np.concatenate((left_gradients, right_gradients), axis=0)
        indices = np.concatenate((left_indices, right_indices))
        gradients, indices = self._gather_indices(gradients, indices)
        updates = self.lr * gradients


        right_updates = self.lr * right_gradients

        # update radius
        self.kv.syn0[indices, 0] -= updates[:,0]
        # update center
        self.kv.syn0[indices, 1:] = self.metric.exp_map(self.kv.syn0[indices, 1:], -updates[:,1:], axis=1)
        # clip into subspace
        self.kv.syn0[indices,:] = self._clip_vectors(self.kv.syn0[indices,:], axis=1)


        if np.abs(self.kv.syn0[0,:]).sum() > 1e20:
            print("##### NaN observed!!")
            print(self.kv.syn0[np.isnan(self.kv.syn0.sum(axis=1))])
            print("#####")
            print(batch.loss_gradients_u)
            raise()

        return loss, pos_loss, neg_loss



    @staticmethod
    def _gather_indices(vectors, indices):
        batch_size, dim = vectors.shape
        new_indices, index_map = np.unique(indices, return_inverse=True)
        new_vectors = np.zeros((new_indices.size, dim))
        for src, dst in enumerate(index_map):
            new_vectors[dst,:] += vectors[src,:]

        return new_vectors, new_indices

    
    def compute_loss(self, left_vectors, right_vectors, labels, return_pos_neg=False):
        """ compute loss

        left_vectors: (n, dim)-array
            vectors for right side hand.

        right_vectors: (n, dim)-array
            vectors for left side hand.

        labels: labels for each pair.
                1: positive, 0: negative
        """

        left_radii = left_vectors[:,0]
        left_centers = left_vectors[:,1:]
        right_radii = right_vectors[:,0]
        right_centers = right_vectors[:,1:]
        dist = self.metric.compute_distance(left_centers, right_centers, axis=1)
        loss_vec = self.loss_function.compute_loss(dist, left_radii, right_radii, labels)

        pos_loss = loss_vec[labels].sum()
        neg_loss = loss_vec[~labels].sum()
        loss = pos_loss + neg_loss

        if return_pos_neg:
            return loss, pos_loss, neg_loss
        else:
            return loss



    def compute_loss_gradients(self, left_vectors, right_vectors, labels):
        """ compute loss gradients

        left_vectors: (n, dim)-array
            vectors for right side hand.

        right_vectors: (n, dim)-array
            vectors for left side hand.

        labels: labels for each pair.
                1: positive, 0: negative

        # returns
        grad_left: gradient for left vectors of the pair
        grad_right: gradient for right vectors of the pair
        """

        labels = np.asarray(labels, dtype=bool)

        # left_indices, right_indices = np.array(pairs).T
        # left_vectors = self.kv.syn0[left_indices]
        # right_vectors = self.kv.syn0[right_indices]

        # (n, dim-1)
        left_centers = left_vectors[:,1:]
        right_centers = right_vectors[:,1:]

        # (n)
        left_radii = left_vectors[:,0]
        right_radii = right_vectors[:,0]
        dist = self.metric.compute_distance(left_centers, right_centers, axis=1)

        # (n)
        loss_diff_dist, loss_diff_left_radii, loss_diff_right_radii = self.loss_function.compute_loss_diff(dist, left_radii, right_radii, labels)

        # (n, dim-1)
        dist_grad_left_centers = self.metric.compute_distance_grad_u(left_centers, right_centers, axis=1)
        dist_grad_right_centers = self.metric.compute_distance_grad_v(left_centers, right_centers, axis=1)

        # (n, dim-1)
        loss_grad_left_centers = loss_diff_dist[:,np.newaxis] * dist_grad_left_centers
        loss_grad_right_centers = loss_diff_dist[:,np.newaxis] * dist_grad_right_centers

        # (n, dim)
        loss_grad_left = np.concatenate((loss_diff_left_radii[:,np.newaxis], loss_grad_left_centers), axis=1)
        loss_grad_right = np.concatenate((loss_diff_right_radii[:,np.newaxis], loss_grad_right_centers), axis=1)

        return loss_grad_left, loss_grad_right
        

class DiskEmbeddingKeyedVectors(DAGEmbeddingKeyedVectors):

    def __init__(self, metric, *args, **kwargs):
        assert isinstance(metric, QuasimetricSpaceBase)
        self.metric = metric

        super().__init__(*args, **kwargs)

    def is_a_scores_vector_batch(self, alpha, parent_vectors, other_vectors, rel_reversed):
        if rel_reversed:
            parent_vectors, other_vectors = other_vectors, parent_vectors

        distances = self.metric.compute_distance(parent_vectors[:,1:], other_vectors[:,1:], axis=1)

        return distances - parent_vectors[:,0] + other_vectors[:,0]

