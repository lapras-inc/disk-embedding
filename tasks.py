import os
import re
import time
import itertools

import luigi
from luigi.util import requires, inherits
import numpy as np
import pandas as pd

from config import myconfig
conf = myconfig()
import utils
import target
from relations import Relations
import evaluations


class PrepareDataset(luigi.ExternalTask):
    """ Pseudo task which points dataset file.
    """

    # Dataset switch
    wn = luigi.Parameter("mammal") # mammal or noun or hep-th-samp
    task_type = luigi.Parameter("0percent")

    def output(self):
        if self.wn in ["hep-th", "hep-th-samp", "r_hep-th", "r_hep-th-samp"]:
            full_data_filepath = os.path.join(conf.input_dir, self.wn + '.tsv')
        else:
            full_data_filepath = os.path.join(conf.input_dir, '{}_closure.tsv'.format(self.wn))

        if self.task_type == 'reconstruction':
            postfix = {
                    "train": '.full_transitive',
                    "valid_pos": '.full_transitive',
                    "valid_neg": '.full_neg',
                    "test_pos": '.full_transitive',
                    "test_neg": '.full_neg',
                    }
        else:
            postfix = {
                    "train": '.train_' + self.task_type,
                    "valid_pos": '.valid',
                    "valid_neg": '.valid_neg',
                    "test_pos": '.test',
                    "test_neg": '.test_neg',
                    }

        return {k: luigi.LocalTarget(full_data_filepath + pf) for k, pf in postfix.items()}



@inherits(PrepareDataset)
class TrainEvalOneModel(luigi.Task):

    ###### Parameters

    # Common Model Parameters
    model_class = luigi.Parameter()
    dim = luigi.IntParameter(5)

    # Common Training Parameters
    init_range_min = luigi.FloatParameter(-0.001)
    init_range_max = luigi.FloatParameter(+0.001)
    lr = luigi.FloatParameter()
    opt = luigi.Parameter("rsgd")
    burn_in = luigi.IntParameter(0)
    batch_size = luigi.IntParameter(10)
    epochs = luigi.IntParameter()
    seed = luigi.IntParameter(0)

    # Negative Samples
    num_negative = luigi.IntParameter(10)
    neg_sampl_strategy = luigi.Parameter("true_neg_non_leaves")
    where_not_to_sample = luigi.Parameter("children")
    neg_edges_attach = luigi.Parameter("parent")
    always_v_in_neg = luigi.BoolParameter(True)
    neg_sampling_power = luigi.FloatParameter(0.75)

    # Pre-training
    use_pretrain = luigi.BoolParameter(False)
    pretrain_params = luigi.DictParameter({})
    pretrain_resc_vecs = luigi.FloatParameter(0.7)

    use_slac = luigi.BoolParameter()

    pretrain_in_less_dim = luigi.BoolParameter(significant=False)

    # Model Specific Parameters
    model_parameters = luigi.DictParameter({})
        # ('epochs_init_burn_in', 20),
        # ('K', 0.1),
        # ('margin', 0.01),
        # ('epsilon', 1e-5),

    def requires(self):
        reqs = {"data": self.clone(PrepareDataset)}

        if self.use_pretrain:
            pretrain_params = dict(self.pretrain_params)
            pretrain_params["wn"] = self.wn
            pretrain_params["task_type"] = self.task_type

            # used only for Disk Embeddings to pretrain with n-1 dimensional symmetric model
            if self.pretrain_in_less_dim:
                pretrain_params["dim"] = self.dim - 1
            else:
                pretrain_params["dim"] = self.dim


            reqs["pretrain"] = TrainEvalOneModel(**pretrain_params)

        return reqs

    def run(self):
        time_start = time.time()

        np.random.seed(self.seed)

        # params = {k: v for k,v in self.get_param_values()}
        params = self.param_kwargs.copy()
        self.output()["params"].dump(params)

        # build and initialize model
        model = self.build_model()

        # supply pretrained vector
        if self.use_pretrain:
            pretrain_vector = self.input()["pretrain"]["vector"].load()

            model.supply_init_vectors(pretrain_vector * self.pretrain_resc_vecs)

        self.logger.info("##################  Start training ###################")
        self.logger.info(self.task_id)

        history = []


        best_valid_f1 = -1
        best_test_f1 = -1
        best_epoch = -1

        for epochs_from in range(0, self.epochs, conf.eval_every_n_epochs):
            model.train(epochs=conf.eval_every_n_epochs,
                        batch_size=self.batch_size,
                        print_every=conf.eval_every_n_epochs)

            epochs_done = epochs_from + conf.eval_every_n_epochs

            self.logger.info("### start eval after {} epochs.".format(epochs_done))
            self.logger.info('MODEL = %s\n' % (self.task_id))

            test_f1, valid_f1, eval_results = self.eval_model(model)

            if valid_f1 > best_valid_f1:
                best_valid_f1 = valid_f1
                best_test_f1 = test_f1
                best_epoch = epochs_done

            history.append({
                "best_epoch": best_epoch,
                "best_test_f1": best_test_f1,
                "best_valid_f1": best_valid_f1,
                "epochs": epochs_done,
                "test_f1": test_f1,
                "valid_f1": valid_f1,
                "result_str": eval_results,
                "time_elapsed": time.time() - time_start,
                })

            self.logger.info('====> best so far f1 test={:.2f}; valid={:.2f} - after {} epochs.'.format(best_test_f1, best_valid_f1, best_epoch))
            self.logger.info("### end eval.")

            # save live scores.
            self.output()["history"].dump(history)

        time_end = time.time()

        self.output()["history"].dump(history)

        if conf.save_trained_model:
            self.output()["model"].dump(model)

        if conf.save_trained_vector:
            self.output()["vector"].dump(model.kv.syn0)

        results = params.copy()
        results.update({
            "best_epoch": best_epoch,
            "best_test_f1": best_test_f1,
            "best_valid_f1": best_valid_f1,
            "time_elapsed": time_end - time_start,
            })
        self.output()["results"].dump(results)

    def root_path(self):
        return os.path.join(conf.output_dir, self.task_name())

    def task_name(self):
        return "noname/" + self.task_id

    def output(self):
        root_path = self.root_path()

        out = {}
        out["log"] = luigi.LocalTarget(os.path.join(root_path, "train.log"))
        out["history"] = target.CsvTarget(os.path.join(root_path, "history.csv"))
        out["params"] = target.JsonTarget(os.path.join(root_path, "params.json"))
        out["results"] = target.JsonTarget(os.path.join(root_path, "results.json"))
        if conf.save_trained_model:
            out["model"] = target.ModelTarget(self.get_model_class(), os.path.join(root_path, "model.dat"))
        if conf.save_trained_vector:
            out["vector"] = target.NpyTarget(os.path.join(root_path, "vector.npy"))
        return out

    @property
    def logger(self):
        if not hasattr(self, "_logger"):
            self._logger = utils.setup_logger(self.output()["log"].path, also_stdout=conf.log_also_stdout)
        return self._logger


    def get_model_class(self):
        if self.model_class == 'PoincareNIPS':
            from models.poincare_model import PoincareModel
            return PoincareModel

        if self.model_class == 'EucSimple':
            from models.eucl_simple_model import EuclSimpleModel
            return EuclSimpleModel

        elif self.model_class == 'OrderEmb':
            from models.order_emb_model import OrderModel
            return OrderModel

        elif self.model_class == 'HypCones':
            from models.hyp_cones_model import HypConesModel
            return HypConesModel

        # elif self.model_class == 'DiskEmb':
            # from models.disk_emb_model import DiskEmbeddingModel
            # return DiskEmbeddingModel

        elif self.model_class == 'DiskEmbOrig':
            from models.disk_emb_model_orig import DiskEmbeddingModel as DiskEmbOrig
            return DiskEmbOrig

        else:
            raise ValueError("Unkown model_class: {}".format(self.model_class))

    def build_model(self):
        train_path = self.input()["data"]["train"].path
        train_data = Relations(train_path, reverse=False)

        cls = self.get_model_class()

        model = cls(train_data=train_data,
                dim=self.dim,
                init_range=(self.init_range_min, self.init_range_max),
                lr=self.lr,
                opt=self.opt,  # rsgd or exp_map
                burn_in=self.burn_in,
                seed=self.seed,

                num_negative=self.num_negative,
                neg_sampl_strategy=self.neg_sampl_strategy,
                where_not_to_sample=self.where_not_to_sample,
                neg_edges_attach=self.neg_edges_attach,
                always_v_in_neg = self.always_v_in_neg,
                neg_sampling_power=self.neg_sampling_power,

                logger=self.logger,

                # model-specific parameters
                **self.model_parameters
                )

        return model

    def eval_model(self, model):
        """ Evaluation as binary classification
        """
        # FIXME ugly condition branches
        if self.model_class == "PoincareNIPS":
            alphas_to_validate = [1000, 100, 30, 10, 3, 1, 0.3, 0.1, 0]
        elif self.model_class == "HypCones":
            # FIXME Invalid use of variable: K is passed instead of alpha.
            alphas_to_validate = [model.K]
        else:
            # FIXME Meaningless value is passed if alpha is not needed.
            alphas_to_validate = [-1e12]
        
        input_data = self.input()["data"]

        # Validation
        eval_result_classif, best_alpha, _, best_test_f1, best_valid_f1 = evaluations.eval_classification(
            logger=self.logger,
            task=self.task_type,
            valid_pos_path=input_data["valid_pos"].path,
            valid_neg_path=input_data["valid_neg"].path,
            test_pos_path=input_data["test_pos"].path,
            test_neg_path=input_data["test_neg"].path,
            vocab=model.kv.vocab,
            score_fn=model.kv.is_a_scores_from_indices,
            alphas_to_validate=alphas_to_validate, # 0 means only distance
        )

        print(eval_result_classif)
        result_str = evaluations.pretty_print_eval_map(eval_result_classif)
        self.logger.info('BEST classification ALPHA = %.3f' % best_alpha)
        self.logger.info(result_str)
        return float(best_test_f1), float(best_valid_f1), result_str


# Specific Tasks
class NamedTrainTask(TrainEvalOneModel):
    def default_params(self):
        raise NotImplementedError()

    def task_name(self):
        # print([self.task_family, dim, self.wn, self.task_type, t.task_id[len(t.task_family)+2:]])
        return "{}/{}_dim{}_{}_{}_epoch{}_{}".format(self.task_family, self.task_family, str(self.dim), self.wn, self.task_type, self.epochs, self.task_id[-10:])

def set_default_params(default_params):
    def _decorator(cls):
        params = dict(cls.get_params())
        for name, value in default_params.items():
            if name in params:
                param_type = type(params[name])
                setattr(cls, name, param_type(value))
        return cls
    return _decorator


class RunAnywayTask(luigi.Task):
    targ = luigi.TaskParameter()
    #try_once = luigi.BoolParameter(False)

    def run(self):
        cls = self.targ
        task = cls()
        task.run()



if __name__ == '__main__':
    luigi.run()


