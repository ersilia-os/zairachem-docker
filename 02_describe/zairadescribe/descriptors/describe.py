import os

from .raw import RawDescriptors

from zairabase import ZairaBase
from zairabase.utils.pipeline import PipelineStep


class Describer(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.output_dir = os.path.abspath(self.path)
        assert os.path.exists(self.output_dir)
        self.logger.debug(self.path)

    def _raw_descriptions(self):
        step = PipelineStep("raw_descriptions", self.output_dir)
        if not step.is_done():
            RawDescriptors().run()
            step.update()

    def run(self):
        self.reset_time()
        self._raw_descriptions()
        self.update_elapsed_time()
