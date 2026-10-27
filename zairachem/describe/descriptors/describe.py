import json
import os
from pathlib import Path

from zairachem.base import ZairaBase
from zairachem.base.generate_config import generate_compose
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.base.utils.utils import install_docker_compose
from zairachem.base.vars import (
  METADATA_SUBFOLDER,
  NETWORK_NAME,
  PARAMETERS_FILE,
  get_free_ports,
)
from zairachem.describe.descriptors import compose
from zairachem.describe.descriptors.raw import RawDescriptors
from zairachem.describe.descriptors.utils import (
  _ensure_network,
  _service_name,
)

cwd = Path(__file__).parent.parent
base_file_path = cwd / "files"
install_file = base_file_path / "install_compose.sh"


class Describer(ZairaBase):
  """Featurize the run's molecules with the selected Ersilia models.

  Streams the compounds through one Docker model server per descriptor model into
  ``descriptors/<model>/raw.h5``. The servers belong to the user's shared compose project and stay
  up between runs. The run's projection models are computed here too, alongside the descriptors.

  Parameters
  ----------
  path : str
    Run folder.
  batch_size : int, optional
    Rows per chunk when reading from or writing to the isaura store (requests to the model
    servers are ``DESCRIBE_BATCH_SIZE`` molecules).
  workers : int, optional
    Most models featurized at once (they are also admitted by memory and CPU).
  """

  def __init__(self, path, batch_size=None, workers=None):
    ZairaBase.__init__(self)
    self.path = path
    self.output_dir = os.path.abspath(self.path)
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir, exist_ok=True)
    self.models = self._get_models()
    self.batch_size = batch_size
    self.workers = workers
    assert os.path.exists(self.output_dir)

  def _get_models(self):
    with open(os.path.join(self.path, METADATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
      data = json.load(f)
    featurizers = data["featurizer_ids"]
    if self.is_predict():
      # Only the descriptors the trained pooled model uses (the --max-descriptors selection when it
      # was screened, less those the pooling left out) — no point featurizing the others.
      from zairachem.base.utils.descriptors import pooled_descriptors

      selected = set(pooled_descriptors(self.get_trained_dir()))
      featurizers = [m for m in featurizers if m in selected]
    return featurizers + data["projection_ids"]

  def create_config_files(self):
    """Add this run's models to the shared compose project, keeping every existing model's port.

    The compose file lists every model any run has used, so a model added now does not change the
    services already running (``compose up`` then leaves them alone).
    """
    with compose.locked("config"):
      ports = compose.load_ports()
      missing = [m for m in self.models if m not in ports]
      if missing:
        taken = set(ports.values())
        fresh = [p for p in get_free_ports(len(missing) + len(taken)) if p not in taken]
        ports.update(zip(missing, fresh))
        compose.save_ports(ports)
      compose_yml_file = compose.compose_file()
      text = generate_compose(dict(sorted(ports.items())))
      if not compose_yml_file.exists() or compose_yml_file.read_text() != text:
        compose_yml_file.parent.mkdir(parents=True, exist_ok=True)
        compose_yml_file.write_text(text)

  def setup_model_servers(self, model_ids=None):
    """Start the redis cache and the servers of ``model_ids`` (default: none), if not already up.

    The run stops the servers it started when it ends (see :mod:`~zairachem.describe.descriptors.compose`).
    Describe starts all of a run's servers at once when it begins (see :meth:`_raw_descriptions`).
    """
    self.create_config_files()
    _ensure_network(NETWORK_NAME)
    if compose.compose_cmd() is None:
      install_docker_compose(install_file)
      compose.compose_cmd.cache_clear()
    services = ["redis"] + [_service_name(m) for m in model_ids or []]
    if not compose.up(services):
      raise RuntimeError(
        "Could not start the model servers; see ~/zairachem/commands.log for the compose output."
      )

  def _raw_descriptions(self):
    step = PipelineStep("raw_descriptions", self.output_dir)
    if not step.is_done():
      # Start every server this run needs at once (compose boots them in parallel); a server only
      # boots its web app here, the model loads per request, so idle servers cost little memory.
      # The scheduler then admits models by memory and CPU, each finding its server already up.
      self.setup_model_servers(self.models)
      RawDescriptors(path=self.path, batch_size=self.batch_size, workers=self.workers).run()
      step.update()
    else:
      self.logger.info("Descriptors already done — skipping.")

  def run(self):
    """Start the model servers and compute the raw descriptors, skipping them if already done."""
    self.reset_time()
    self._raw_descriptions()
    self.update_elapsed_time()
