from tasks import NamedTrainTask, set_default_params

@set_default_params({
    'model_class': 'PoincareNIPS',

    'epochs': 100,
    'init_range_min': -0.0001,
    'init_range_max': 0.0001,
    'lr': 0.03,
    'opt': 'exp_map',
    'burn_in': 20,

    'num_negative': 10,
    'neg_sampl_strategy': 'true_neg',
    'where_not_to_sample': 'ancestors',
    'neg_edges_attach': 'child',

    'always_v_in_neg': True,
    'neg_sampling_power': 0.75,

    'model_parameters': {
        'loss_type': 'nll',
        'epsilon': 1e-05,
        'maxmargin_margin': 1.0,
        'neg_r': 2.0,
        'neg_t': 1.0,
        'neg_mu': 1.0,
        }
    })
class PoincareNIPSTask(NamedTrainTask):
    """ Poincare embeddings of Nickel et al., NIPS'18
    """
    pass



@set_default_params({
    'model_class': 'OrderEmb',

    'epochs': 500,
    'init_range_min': -0.1,
    'init_range_max': +0.1,
    'lr': 0.1,
    'opt': 'sgd',
    'burn_in': 0,

    'num_negative': 10,
    'neg_sampl_strategy': 'true_neg',
    'where_not_to_sample': 'children',
    'neg_edges_attach': 'parent',

    'always_v_in_neg': False,
    'neg_sampling_power': 0,

    'model_parameters': {
        'margin': 1
        },
    })
class OrderEmbTask(NamedTrainTask):
    """ Vendrov et. al. 2016
    """
    pass

        

# Poincare Embedding (Nickel et. al. 2017) for pre-training HypCones and HypDiskEmb.
pretrain_poincare_params = {
        'model_class': 'PoincareNIPS',

        'init_range_min': -0.0001,
        'init_range_max': 0.0001,
        'lr': 0.03,
        'opt': 'rsgd',
        'burn_in': 20,

        'epochs': 100,

        'num_negative': 10,
        'neg_sampl_strategy': 'true_neg',
        'where_not_to_sample': 'children',
        'neg_edges_attach': 'parent',

        'always_v_in_neg': True,
        'neg_sampling_power': 0.75,

        'model_parameters': {
            'loss_type': 'nll',
            'epsilon': 1e-05,
            'maxmargin_margin': 1.0,
            'neg_r': 2.0,
            'neg_t': 1.0,
            'neg_mu': 1.0,
            },
        }

@set_default_params({
    'model_class': 'HypCones',

    'epochs': 300,
    'init_range_min': -0.1,
    'init_range_max': +0.1,
    'lr': 0.0003,
    'opt': 'rsgd',
    'burn_in': 0,

    'num_negative': 10,
    'neg_sampl_strategy': 'true_neg',
    'where_not_to_sample': 'children',
    'neg_edges_attach': 'parent',

    'always_v_in_neg': False,
    'neg_sampling_power': 0.0,

    'model_parameters': {
        'margin': 0.01,
        'K': 0.1,
        'epsilon': 1e-05,
        },

    'use_pretrain': True,
    'pretrain_resc_vecs': 0.7,
    'pretrain_params': pretrain_poincare_params,
    })
class HypConesTask(NamedTrainTask):
    """ Ganea et. al. 2018
    """
    pass



@set_default_params({
    'model_class': 'DiskEmbOrig',

    'epochs': 300,
    'init_range_min': -0.1,
    'init_range_max': +0.1,
    'lr': 0.03,
    'opt': 'disk_rsgd',
    'burn_in': 0,

    'num_negative': 10,
    'neg_sampl_strategy': 'all',
    'neg_edges_attach': 'both',

    'model_parameters': {
        'metric': "sphere",
        'loss': "relu_margin",
        'margin': 0.01,
        'init_vector_k': 0.1,
        'init_vector_conv': "poincare",
        },

    'use_pretrain': True,
    'pretrain_resc_vecs': 0.7,
    'pretrain_params': pretrain_poincare_params,
    })
class SphericalDiskEmbTask(NamedTrainTask):
    """ Our Disk Embedding in the sphere
    """
    pass


@set_default_params({
    'model_class': 'DiskEmbOrig',

    'epochs': 300,
    'init_range_min': -0.1,
    'init_range_max': +0.1,
    'lr': 0.03,
    'opt': 'disk_rsgd',
    'burn_in': 0,

    'num_negative': 10,
    'neg_sampl_strategy': 'all',
    'neg_edges_attach': 'both',

    'pretrain_in_less_dim': True,
    'model_parameters': {
        'metric': "hyp",
        'loss': "relu_margin",
        'margin': 0.5,
        'init_vector_conv': "poincare2lorentz",
        },

    'use_pretrain': True,
    'pretrain_resc_vecs': 0.7,
    'pretrain_params': pretrain_poincare_params,
    })
class HypDiskEmbTask(NamedTrainTask):
    """ Our Disk Embedding in the hyperbolic space
    """
    pass


# Simple Euclidean embedding for pre-training EucDiskEmb

pretrain_euc_params = {
        'model_class': '',

        'init_range_min': -0.0001,
        'init_range_max': 0.0001,
        'lr': 0.03,
        'opt': 'rsgd',
        'burn_in': 20,

        'epochs': 100,

        'num_negative': 10,
        'neg_sampl_strategy': 'true_neg',
        'where_not_to_sample': 'children',
        'neg_edges_attach': 'parent',

        'always_v_in_neg': True,
        'neg_sampling_power': 0.75,

        'model_parameters': {
            'loss_type': 'nll',
            'epsilon': 1e-05,
            'maxmargin_margin': 1.0,
            'neg_r': 2.0,
            'neg_t': 1.0,
            'neg_mu': 1.0,
            },
        }
@set_default_params({
    'model_class': 'DiskEmbOrig',

    'epochs': 500,
    'init_range_min': -0.1,
    'init_range_max': +0.1,
    'lr': 0.2,
    'opt': 'disk_rsgd',
    'burn_in': 0,

    'num_negative': 10,
    'neg_sampl_strategy': 'all',
    'neg_edges_attach': 'both',

    'always_v_in_neg': False,
    'neg_sampling_power': 0,

    'model_parameters': {
        'loss': "relu_margin",
        'margin': 1.0,
        'metric': "euc",
        },
    })
class EucDiskEmbTask(NamedTrainTask):
    """ Our Disk Embedding in the Euclidean Space
    """
    pass

