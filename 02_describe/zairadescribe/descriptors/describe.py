import os

from .treated import TreatedDescriptors
from .manifolds import Manifolds
from .raw import RawDescriptors

from zairabase import ZairaBase
from zairabase.utils.pipeline import PipelineStep


class Describer(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
            print(self.path)
        else:
            self.path = path
            print(self.path)
        self.output_dir = os.path.abspath(self.path)
        assert os.path.exists(self.output_dir)
        self.logger.debug(self.path)

    def _raw_descriptions(self):
        step = PipelineStep("raw_descriptions", self.output_dir)
        if not step.is_done():
            RawDescriptors().run()
            step.update()

    def _treated_descriptions(self):
        step = PipelineStep("treated_descriptions", self.output_dir)
        if not step.is_done():
            TreatedDescriptors().run()
            step.update()

    def _manifolds(self):
        step = PipelineStep("manifolds", self.output_dir)
        if not step.is_done():
            Manifolds().run()
            step.update()

    def run(self):
        self.reset_time()
        self._raw_descriptions()
        self._treated_descriptions()
        self._manifolds()
        self.update_elapsed_time()