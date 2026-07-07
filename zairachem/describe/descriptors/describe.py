import json, os
from zairachem.describe.descriptors.raw import RawDescriptors
from zairachem.describe.descriptors.utils import (
  service_exists,
  write_service_file,
  _ensure_network,
  _recreate_container_if_exists,
)
from zairachem.base.utils.utils import install_docker_compose
from zairachem.base.utils.terminal import run_command
from zairachem.base import ZairaBase
from zairachem.base.utils.pipeline import PipelineStep
from zairachem.base.generate_config import generate_compose_and_nginx
from zairachem.base.vars import (
  NETWORK_NAME,
  METADATA_SUBFOLDER,
  PARAMETERS_FILE,
  ALL_FEATURIZER,
  get_free_ports,
)
from pathlib import Path


cwd = Path(__file__).parent.parent
base_file_path = cwd / "files"
base_config_path = base_file_path / "configs"
nginx_config_file = base_config_path / "nginx.conf"
compose_yml_file = base_config_path / "docker-compose.yml"
install_file = base_file_path / "install_compose.sh"


class Describer(ZairaBase):
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

  def _get_models_ports(self):
    return dict(zip(self.models, get_free_ports(len(self.models))))

  def _get_models(self):
    with open(os.path.join(self.path, METADATA_SUBFOLDER, PARAMETERS_FILE), "r") as f:
      data = json.load(f)
    featurizers = data["featurizer_ids"]
    if self.is_predict():
      # Only the descriptors the trained model actually uses (the --max-descriptors selection when it
      # was screened, else all) — no point featurizing descriptors the pooled model will ignore.
      from zairachem.base.utils.descriptors import effective_descriptors

      selected = set(effective_descriptors(self.get_trained_dir()))
      featurizers = [m for m in featurizers if m in selected]
    return featurizers + data["projection_ids"]

  def create_config_files(self):
    all_service_exists = service_exists(compose_yml_file, self.models)

    if isinstance(all_service_exists, bool) and not all_service_exists:
      os.remove(compose_yml_file)
      os.remove(nginx_config_file)

    if not os.path.exists(compose_yml_file) or not os.path.exists(nginx_config_file):
      os.makedirs(base_config_path, exist_ok=True)
      compose, nginx_conf = generate_compose_and_nginx(self._get_models_ports())
      Path(compose_yml_file).write_text(compose)
      Path(nginx_config_file).write_text(nginx_conf)

  def setup_model_servers(self):
    self.create_config_files()
    _ensure_network(NETWORK_NAME)
    _recreate_container_if_exists()
    install_docker_compose(install_file)
    try:
      run_command(["docker-compose", "-f", os.fspath(compose_yml_file), "up", "-d"], quiet=True)
    except Exception as e:
      self.logger.warning(f"[describe] docker-compose up failed: {e}")

  def _raw_descriptions(self):
    step = PipelineStep("raw_descriptions", self.output_dir)
    if not step.is_done():
      RawDescriptors(path=self.path, batch_size=self.batch_size, workers=self.workers).run()
      step.update()
    else:
      self.logger.info("Descriptors already done — skipping.")

  def run(self):
    self.setup_model_servers()
    write_service_file(ALL_FEATURIZER)
    self.reset_time()
    self._raw_descriptions()
    self.update_elapsed_time()
