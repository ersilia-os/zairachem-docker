import re
from zairachem.base.vars import REDIS_IMAGE, NETWORK_NAME


def _sanitize(name):
  s = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")
  return s or "svc"


def _service_block(model_id, host_port, network_name):
  service_name = f"{_sanitize(model_id)}_api"
  image_name = f"ersiliaos/{model_id.lower()}"
  return f"""  {service_name}:
    image: {image_name}
    environment:
      REDIS_HOST: redis
      REDIS_PORT: "6379"
      REDIS_URI: "redis://redis:6379"
      REDIS_EXPIRATION: "604800"
    ports:
      - "127.0.0.1:{host_port}:80"
    networks:
      - {network_name}
    depends_on:
      redis:
        condition: service_healthy
"""


def _networks_block(
  network_key,
  *,
  docker_network_name=None,
  ipam_subnet=None,
  driver="bridge",
  external=False,
):
  name = docker_network_name or network_key
  if external:
    return f"""networks:
  {network_key}:
    external: true
    name: {name}

"""
  ipam = (
    f"""
    ipam:
      config:
        - subnet: {ipam_subnet}"""
    if ipam_subnet
    else ""
  )
  return f"""networks:
  {network_key}:
    external: true
    name: {name}
    driver: {driver}{ipam}

"""


def generate_compose(
  models_with_ports,
  network_name=NETWORK_NAME,
  *,
  docker_network_name=None,
  ipam_subnet=None,
  external=False,
):
  """The docker-compose YAML for one run: a redis cache plus one API service per model.

  Redis is stateless (no volume), so it disappears with the run's compose project.
  """
  header = "services:\n"

  redis = f"""  redis:
      image: {REDIS_IMAGE}
      command:
        - redis-server
        - --save
        - ""
        - --appendonly
        - "no"
        - --maxmemory
        - 4gb
        - --maxmemory-policy
        - allkeys-lru
        - --maxmemory-samples
        - "10"
      healthcheck:
        test: ["CMD", "redis-cli", "ping"]
        interval: 5s
        timeout: 3s
        # redis answers `ping` almost immediately once up; 5×5s is ample headroom. (Was 20, i.e. up to
        # ~100s, which let a slow/missing redis stall every model service's `depends_on` for minutes.)
        retries: 5
      networks:
        - {network_name}
"""

  services = "".join(
    _service_block(model_id, port, network_name)
    for model_id, port in sorted(models_with_ports.items())
  )

  networks = _networks_block(
    network_name,
    docker_network_name=docker_network_name,
    ipam_subnet=ipam_subnet,
    driver="bridge",
    external=external,
  )

  return header + redis + services + networks
