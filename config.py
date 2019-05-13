import luigi

class myconfig(luigi.Config):
    eval_every_n_epochs = luigi.IntParameter(10)
    seed = luigi.IntParameter(0)

    output_dir = luigi.Parameter("./data/output")
    input_dir = luigi.Parameter("./data/maxn")
    log_also_stdout = luigi.BoolParameter(True)
    save_trained_model = luigi.BoolParameter(False)
    save_trained_vector = luigi.BoolParameter(True)

