import luigi
import named_tasks as named
import pandas as pd
import target

class _GatherResults(luigi.Task):


    def run(self):
        
        results = []
        
        for req in self.requires():
            if not req.complete():
                continue
            inp = req.output()
            dct = inp["results"].load()
            dct.update(dct["model_parameters"])
            del dct["model_parameters"]
            results.append(dct)
            
        results = pd.DataFrame(results).fillna("-")
        piv = results.pivot_table(index=("model_class", "metric", "loss"), columns=("dim", "task_type"), values="best_test_f1").sort_index()

        print(piv)
        self.output().dump(piv)


    def output(self):
        return target.CsvTarget(self.filename())

    def filename(self):
        raise NotImplementedError()

class WordNetNounTask(_GatherResults):
    wn = luigi.Parameter("noun")

    def filename(self):
        return "./data/results_wn_noun.csv"

    def requires(self):
        classes = [
                named.HypConesTask,
                named.PoincareNIPSTask,
                named.OrderEmbTask,
                named.EucDiskEmbTask,
                named.SphericalDiskEmbTask,
                named.HypDiskEmbTask
                ]
        for cls in classes:
            for dim in [5,10]:
                for task_type in ["0percent", "10percent", "25percent", "50percent"]:
                    for seed in range(1):
                        yield self.clone(cls, dim=dim, task_type=task_type, seed=seed)


class WordNetNounRevTask(_GatherResults):
    wn = luigi.Parameter("r_noun")

    def filename(self):
        return "./data/results_wn_noun_rev.csv"

    def requires(self):
        classes = [
                named.HypConesTask,
                named.PoincareNIPSTask,
                named.OrderEmbTask,
                named.EucDiskEmbTask,
                named.SphericalDiskEmbTask,
                named.HypDiskEmbTask
                ]

        for cls in classes:
            for dim in [5,10]:
                for task_type in ["0percent", "10percent", "25percent", "50percent"]:
                    for seed in range(1):
                        yield self.clone(cls, dim=dim, task_type=task_type, seed=seed)

class RunAll(luigi.WrapperTask):

    def requires(self):
        yield WordNetNounTask()
        yield WordNetNounRevTask()


if __name__ == '__main__':
    luigi.run()

