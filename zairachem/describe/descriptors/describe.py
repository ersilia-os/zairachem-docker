import os
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
  ERSILIA_HUB_DEFAULT_MODELS_WITH_PORT,
  ERSILIA_HUB_DEFAULT_MODELS,
  NETWORK_NAME,
)
from pathlib import Path


cwd = Path(__file__).parent.parent
base_file_path = cwd / "files"
base_config_path = base_file_path / "configs"
nginx_config_file = base_config_path / "nginx.conf"
compose_yml_file = base_config_path / "docker-compose.yml"
install_file = base_file_path / "install_compose.sh"


class Describer(ZairaBase):
  def __init__(self, path):
    ZairaBase.__init__(self)
    if path is None:
      self.path = self.get_output_dir()
    else:
      self.path = path
    self.output_dir = os.path.abspath(self.path)
    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir, exist_ok=True)
    assert os.path.exists(self.output_dir)

  def create_config_files(self):
    all_service_exists = service_exists(compose_yml_file, ERSILIA_HUB_DEFAULT_MODELS)

    if isinstance(all_service_exists, bool) and not all_service_exists:
      os.remove(compose_yml_file)
      os.remove(nginx_config_file)

    if not os.path.exists(compose_yml_file) or not os.path.exists(nginx_config_file):
      os.makedirs(base_config_path, exist_ok=True)
      compose, nginx_conf = generate_compose_and_nginx(ERSILIA_HUB_DEFAULT_MODELS_WITH_PORT)
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
      print(e)

  def _raw_descriptions(self):
    step = PipelineStep("raw_descriptions", self.output_dir)
    if not step.is_done():
      RawDescriptors().run()
      step.update()
    else:
      self.logger.warning(
        "[yellow]Descriptor setup for requested inferece is already done. Skipping this step![/]"
      )

  def run(self):
    self.setup_model_servers()
    write_service_file(ERSILIA_HUB_DEFAULT_MODELS)
    self.reset_time()
    self._raw_descriptions()
    self.update_elapsed_time()
