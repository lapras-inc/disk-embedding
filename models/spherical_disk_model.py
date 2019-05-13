#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Disk Embedding into n-Sphire
"""

from dag_emb_model import DAGEmbeddingModel, DAGEmbeddingBatch, DAGEmbeddingKeyedVectors

import numpy as np

try:
    from autograd import grad  # Only required for optionally verifying gradients while training
    from autograd import numpy as grad_np
    AUTOGRAD_PRESENT = True
except ImportError:
    AUTOGRAD_PRESENT = False

# Cosine clipping epsilon
EPS = 1e-7

class SphericalDiskModel(DAGEmbeddingModel):
    """Class for training, using and evaluating Order Embeddings."""
    # def __init__(*args, **kwargs):
        # super().__init__(*args, **kwargs)


    def _clip_vectors(self, vectors):
        """Clip vectors to have a norm of less than 1 - eps and more than inner_radius + eps.

        Parameters
        ----------
        vectors : numpy.array
            Can be 1-D,or 2-D (in which case the norm for each row is checked).

        Returns
        -------
        numpy.array
            Array clipped in the unit sphere
        """

        vectors = np.copy(vectors)
        
        one_d = len(vectors.shape) == 1
        if one_d:
            norm = np.linalg.norm(vectors[1:]) # first axis is a radius of a disk.
            vectors[1:] /= norm
        else:
            norms = np.linalg.norm(vectors[:,1:], axis=1)
            vectors[:,1:] /= norm

        return vectors


    ### For autograd
    # def _loss_fn(self, matrix, rels_reversed):
        # """Given a numpy array with vectors for u, v and negative samples, computes loss value.

        # Parameters
        # ----------
        # matrix : numpy.array
            # Array containing vectors for u, v and negative samples, of shape (2 + negative_size, dim).
        # rels_reversed : bool

        # Returns
        # -------
        # float
            # Computed loss value.

        # Warnings
        # --------
        # Only used for autograd gradients, since autograd requires a specific function signature.
        # """
        # vector_u = matrix[0]
        # vectors_v = matrix[1:]

        # norm_u = grad_np.linalg.norm(vector_u)
        # norms_v = grad_np.linalg.norm(vectors_v, axis=1)
        # euclidean_dists = grad_np.linalg.norm(vector_u - vectors_v, axis=1)
        # dot_prod = (vector_u * vectors_v).sum(axis=1)

        # if not rels_reversed:
            # # u is x , v is y
            # cos_angle_child = (dot_prod * (1 + norm_u ** 2) - norm_u ** 2 * (1 + norms_v ** 2)) /\
                              # (norm_u * euclidean_dists * grad_np.sqrt(1 + norms_v ** 2 * norm_u ** 2 - 2 * dot_prod))
            # angles_psi_parent = grad_np.arcsin(self.K * (1 - norm_u**2) / norm_u) # scalar
        # else:
            # # v is x , u is y
            # cos_angle_child = (dot_prod * (1 + norms_v ** 2) - norms_v **2 * (1 + norm_u ** 2) ) /\
                              # (norms_v * euclidean_dists * grad_np.sqrt(1 + norms_v**2 * norm_u**2 - 2 * dot_prod))
            # angles_psi_parent = grad_np.arcsin(self.K * (1 - norms_v**2) / norms_v) # 1 + neg_size

        # # To avoid numerical errors
        # clipped_cos_angle_child = grad_np.maximum(cos_angle_child, -1 + EPS)
        # clipped_cos_angle_child = grad_np.minimum(clipped_cos_angle_child, 1 - EPS)
        # angles_child = grad_np.arccos(clipped_cos_angle_child)  # 1 + neg_size

        # energy_vec = grad_np.maximum(0, angles_child - angles_psi_parent)
        # positive_term = energy_vec[0]
        # negative_terms = energy_vec[1:]
        # return positive_term + grad_np.maximum(0, self.margin - negative_terms).sum()


class SphericalDiskBatch(DAGEmbeddingBatch):
    """Compute gradients and loss for a training batch."""
    def __init__(self,
                 vectors_u, # (1, dim, batch_size)
                 vectors_v, # (1 + neg_size, dim, batch_size)
                 indices_u,
                 indices_v,
                 rels_reversed,
                 model: SphericalDiskModel):
        super().__init__(
            vectors_u=vectors_u,
            vectors_v=vectors_v,
            indices_u=indices_u,
            indices_v=indices_v,
            rels_reversed=rels_reversed,
            dag_embedding_model=None)
        self.margin = model.margin


    def _compute_loss(self):
        """Compute and store loss value for the given batch of examples."""
        if self._loss_computed:
            return
        self._loss_computed = True

        self.vectors_u # (1, dim, batch_size)
        self.vectors_v # (1 + neg, dim, batch_size)

        # D
        self.distance_between_centers = np.linalg.norm(self.vectors_u[:,:1,:] - self.vectors_v[:,1:,:], axis=1) # (1 + neg_size, batch_size)

        if not self.rels_reversed:
            # u is x , v is y
            self.radius_child = self.vectors_u[:,0,:] # (1 + neg_size, batch_size)
            self.radius_parent = self.vectors_v[:,0,:] # (1, batch_size)

        else:
            # v is x , u is y
            self.radius_parent = self.vectors_u[:,0,:] # (1 + neg_size, batch_size)
            self.radius_child = self.vectors_v[:,0,:] # (1, batch_size)

        self.kuikomi = self.radius_child - self.radius_parent + self.distance_between_centers
        self.energy_vec = np.maximum(0, self.kuikomi) # (1 + neg_size, batch_size)

        self.pos_loss = self.energy_vec[0].sum()
        self.neg_loss = np.maximum(0, self.margin - self.energy_vec[1:]).sum()
        self.loss = self.pos_loss + self.neg_loss


    def _compute_loss_gradients(self):
        """Compute and store gradients of loss function for all input vectors."""
        if self._loss_gradients_computed:
            return
        self._compute_loss()

        # (1 + neg_size, dim, batch_size)
        _is_radius = np.arange(self.vector_u.shape[1])[np.newaxis,:,np.newaxis] == 0

        self.vector_u_without_radius = np.where(_is_radius[np.newaxis,:,np.newaxis], 0, self.vector_u) # (1, dim, batch_size)
        self.vector_v_without_radius = np.where(_is_radius[np.newaxis,:,np.newaxis], 0, self.vector_v) # (1 + neg_size, dim, batch_size)

        self.normalized_u = self.vector_u_without_radius / np.linalg.norm(self.vector_u_without_radius) # (1, dim, batch_size)
        self.normalized_v = self.vector_v_without_radius / np.linalg.norm(self.vector_v_without_radius) # (1 + neg_size, dim, batch_size)
        self.difference_of_centers = self.normalized_u - self.normalized_v

        # (1 + neg_size, dim, batch_size)
        self.distance_grad_u = np.where(self.distance_between_centers[:,np.newaxis,:] > EPS, self.difference_of_centers, 0)
        self.distance_grad_v = - self.distance_grad_u

        # (1 + neg_size, 1, batchsize)
        self.distance_grad_u_radial = np.sum(self.normalized_u * self.distance_grad_u, axis=1, keepdims=True)
        self.distance_grad_v_radial = np.sum(self.normalized_v * self.distance_grad_v, axis=1, keepdims=True)
 
        # (1 + neg_size, dim, batchsize)
        self.distance_grad_u_projected = self.distance_grad_u - self.normalized_u * self.distance_grad_u_radial
        self.distance_grad_v_projected = self.distance_grad_v - self.normalized_v * self.distance_grad_v_radial


        if not self.rels_reversed:
            # u is x , v is y
            self.radius_child = self.vectors_u[:,0,:] # (1 + neg_size, batch_size)
            self.radius_parent = self.vectors_v[:,0,:] # (1, batch_size)

            self.radius_child_grad_u = np.where(_is_radius[np.newaxis,:,np.newaxis], 1.0, 0.0)
            self.radius_child_grad_v = np.where(_is_radius[np.newaxis,:,np.newaxis], 0.0, 0.0)

            self.radius_parent_grad_u = np.where(_is_radius[np.newaxis,:,np.newaxis], 0.0, 0.0)
            self.radius_parent_grad_v = np.where(_is_radius[np.newaxis,:,np.newaxis], 1.0, 0.0)


        else:
            # v is x , u is y
            self.radius_parent = self.vectors_u[:,0,:] # (1 + neg_size, batch_size)
            self.radius_child = self.vectors_v[:,0,:] # (1, batch_size)

            self.radius_parent_grad_u = np.where(_is_radius[np.newaxis,:,np.newaxis], 1.0, 0.0)
            self.radius_parent_grad_v = np.where(_is_radius[np.newaxis,:,np.newaxis], 0.0, 0.0)

            self.radius_child_grad_u = np.where(_is_radius[np.newaxis,:,np.newaxis], 0.0, 0.0)
            self.radius_child_grad_v = np.where(_is_radius[np.newaxis,:,np.newaxis], 1.0, 0.0)


        self.kuikomi = self.radius_child - self.radius_parent + self.distance_between_centers

        # (1 + neg_size, dim, batch_size)
        self.energy_vec_grad_u = np.where(
                self.kuikomi > 0,
                self.distance_grad_u_projected - self.radius_parent_grad_u + self.radius_child_grad_u,
                np.zeros_like(vectors_u.shape)
                )

        # (1 + neg_size, dim, batch_size)
        self.energy_vec_grad_v = np.where(
                self.kuikomi > 0,
                self.distance_grad_v_projected - self.radius_parent_grad_v + self.radius_child_grad_v,
                np.zeros_like(vectors_v.shape)
                )

        _is_pos = np.arange(self.vectors_v.shape[0]) == 0

        self.pos_loss_grad_u = self.energy_vec_grad_u
        self.pos_loss_grad_v = self.energy_vec_grad_v
        self.neg_loss_grad_u = np.where(self.kuikomi < margin, -self.energy_vec_grad_u)
        self.neg_loss_grad_v = np.where(self.kuikomi < margin, -self.energy_vec_grad_v)

        # (dim, batch_size)
        self.loss_grad_u = np.where(
                _is_pos[:,np.newaxis,np.newaxis],
                self.pos_loss_grad_u,
                self.neg_loss_grad_u
                ).sum(axis=0)

        # (1 + neg, dim, batch_size)
        self.loss_grad_v = np.where(
                _is_pos[:,np.newaxis,np.newaxis],
                self.pos_loss_grad_v,
                self.neg_loss_grad_v
                )

class SphericalDiskKeyedVectors(DAGEmbeddingKeyedVectors):
    """Class to contain vectors and vocab for the :class:`~HypConesModel` training class.
    Used to perform operations on the vectors such as vector lookup, distance etc.
    Inspired from KeyedVectorsBase.
    """
    def __init__(self):
        super(SphericalDisk, self).__init__()

    # def is_a_scores_vector_batch(self, K, parent_vectors, other_vectors, rel_reversed):
        # norm_parent = np.linalg.norm(parent_vectors, axis=1)
        # norm_parent_sq = norm_parent ** 2
        # norms_other = np.linalg.norm(other_vectors, axis=1)
        # norms_other_sq = norms_other ** 2
        # euclidean_dists = np.maximum(np.linalg.norm(parent_vectors - other_vectors, axis=1), 1e-6) # To avoid the fact that parent can be equal to child for the reconstruction experiment
        # dot_prods = (parent_vectors * other_vectors).sum(axis=1)
        # g = 1 + norm_parent_sq * norms_other_sq - 2 * dot_prods
        # g_sqrt = np.sqrt(g)

        # if not rel_reversed:
            # # parent = x , other = y
            # child_numerator = dot_prods * (1 + norm_parent_sq) - norm_parent_sq * (1 + norms_other_sq)
            # child_numitor = euclidean_dists * norm_parent * g_sqrt
            # angles_psi_parent = np.arcsin(K * (1 - norm_parent_sq) / norm_parent)
        # else:
            # # parent = y , other = x
            # child_numerator = dot_prods * (1 + norms_other_sq) - norms_other_sq * (1 + norm_parent_sq)
            # child_numitor = euclidean_dists * norms_other * g_sqrt
            # angles_psi_parent = np.arcsin(K * (1 - norms_other_sq) / norms_other)

        # cos_angles_child = child_numerator / child_numitor
        # assert not np.isnan(cos_angles_child).any()
        # clipped_cos_angle_child = np.maximum(cos_angles_child, -1 + EPS)
        # clipped_cos_angle_child = np.minimum(clipped_cos_angle_child, 1 - EPS)
        # angles_child = np.arccos(clipped_cos_angle_child)  # (1 + neg_size, batch_size)

        # # return angles_child # np.maximum(1, angles_child / angles_psi_parent)
        # return np.maximum(0, angles_child - angles_psi_parent)

