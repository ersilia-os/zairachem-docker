from ersilia import ErsiliaModel


class ModelArtifact(object):
    def __init__(self, model_id):
        self.model_id = model_id
        try:
            self.load_model()
        except:
            self.model = None

    def load_model(self):
        self.model = ErsiliaModel(
            model=self.model_id,
            save_to_lake=False,
            service_class="pulled_docker",
            fetch_if_not_available=True,
            verbose = True
        )

    def run(self,input_csv, output_h5):
        self.model.serve()
        self.model.run(input=input_csv, output=output_h5)
        self.model.close()
    
    def info(self):
        info = self.model.info()
        return info



m = ModelArtifact("eos4u6p")
m.run("../../zaira-chem-docker-tdc/data/AMES_train.csv", "test_ames_eos4u6'.csv")