import re
from zairachem.base.vars import REDIS_IMAGE, NETWORK_NAME, NGINX_HOST_PORT
from typing import Dict


def _sanitize(name: str) -> str:
  s = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")
  return s or "svc"


def _service_block(model_id: str, host_port: int, network_name: str) -> str:
  service_name = f"{_sanitize(model_id)}_api"
  image_name = f"ersiliaos/{model_id.lower()}"
  return f"""  {service_name}:
    image: {image_name}
    environment:
      REDIS_HOST: redis
      REDIS_PORT: "6379"
      REDIS_URI: "redis://redis:6379"
      REDIS_EXPIRATION: "604800"
    restart: unless-stopped
    ports:
      - "{host_port}:80"
    networks:
      - {network_name}
    depends_on:
      - redis
"""


def _nginx_upstream_and_location(model_id: str) -> str:
  service_name = f"{_sanitize(model_id)}_api"
  public_path = f"/{model_id}/"
  return f"""    upstream {service_name} {{
        server {service_name}:80;
        keepalive 64;
    }}

    location {public_path} {{
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_connect_timeout 30s;
        proxy_send_timeout 120s;
        proxy_read_timeout 300s;
        proxy_buffering on;
        proxy_buffers 32 16k;
        proxy_busy_buffers_size 64k;
        proxy_max_temp_file_size 0;
        proxy_next_upstream error timeout http_500 http_502 http_503 http_504;
        proxy_cache api_cache;
        proxy_cache_revalidate on;
        proxy_cache_use_stale error timeout http_500 http_502 http_503 http_504 updating;
        proxy_cache_bypass $http_cache_control $http_pragma;
        proxy_no_cache $http_cache_control $http_pragma;
        proxy_cache_valid 200 301 302 10m;
        proxy_cache_valid 404 1m;
        add_header X-Cache-Status $upstream_cache_status always;
        limit_req zone=perip burst=20 nodelay;
        limit_conn perip_conn 40;
        proxy_pass http://{service_name}/;
    }}
"""


def _networks_block(
  network_key: str,
  *,
  docker_network_name: str | None,
  ipam_subnet: str | None,
  driver: str = "bridge",
  external: bool = False,
) -> str:
  """
  Build the 'networks:' section. If external=True, Compose will attach to an existing
  Docker network with 'name', and we must not specify driver/ipam here.
  """
  if external:
    return f"""networks:
  {network_key}:
    external: true
    name: {NETWORK_NAME}

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
    name: {NETWORK_NAME}
    driver: {driver}{ipam}

"""


def generate_compose_and_nginx(
  models_with_ports: dict[str, int],
  nginx_host_port: int = NGINX_HOST_PORT,
  network_name: str = NETWORK_NAME,
  *,
  docker_network_name: str | None = None,  
  ipam_subnet: str | None = None,         
  external: bool = False,                 
) -> tuple[str, str]:
  header = 'version: "3.9"\nservices:\n'

  redis = f"""  redis:
    container_name: redis
    image: {REDIS_IMAGE}
    command: ["redis-server", "--appendonly", "yes"]
    restart: unless-stopped
    ports:
      - "6379:6379" 
    volumes:
      - redis_data:/data
    networks:
      - {network_name}
"""

  nginx = f"""  nginx:
    image: nginx:alpine
    depends_on:
"""
  for mid in sorted(models_with_ports):
    nginx += f"      - {_sanitize(mid)}_api\n"
  nginx += f"""    ports:
      - "{nginx_host_port}:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - nginx_cache:/var/cache/nginx
    restart: unless-stopped
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

  volumes = """volumes:
  redis_data:
  nginx_cache:
"""

  compose_yaml = header + redis + nginx + services + networks + volumes

  nginx_top = """worker_processes auto;

events {
    worker_connections 4096;
    multi_accept on;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    sendfile        on;
    tcp_nopush      on;
    tcp_nodelay     on;
    server_tokens   off;

    log_format main_json escape=json
      '{'
        '"time_local":"$time_local",'
        '"remote_addr":"$remote_addr",'
        '"request":"$request",'
        '"status":$status,'
        '"bytes_sent":$bytes_sent,'
        '"request_time":$request_time,'
        '"upstream_response_time":"$upstream_response_time",'
        '"upstream_addr":"$upstream_addr",'
        '"cache":"$upstream_cache_status"'
      '}';
    access_log /var/log/nginx/access.log main_json;

    gzip on;
    gzip_comp_level 5;
    gzip_min_length 1024;
    gzip_vary on;
    gzip_proxied any;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml
        application/xhtml+xml
        application/rss+xml
        font/woff
        font/woff2;

    proxy_cache_path /var/cache/nginx/api
        levels=1:2
        keys_zone=api_cache:20m
        max_size=1g
        inactive=10m
        use_temp_path=off;

    limit_req_zone $binary_remote_addr zone=perip:10m rate=10r/s;
    limit_conn_zone $binary_remote_addr zone=perip_conn:10m;

    map $http_upgrade $connection_upgrade {
        default upgrade;
        ''      close;
    }

    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options SAMEORIGIN always;
    add_header Referrer-Policy strict-origin-when-cross-origin always;

    server {
        listen 80;
        server_name _;

        location = / {
            return 200 'Ersilia API gateway is up. Try one of the /<model_id>/ paths.\\n';
            add_header Content-Type text/plain;
        }

"""
  nginx_blocks = "".join(
    _nginx_upstream_and_location(model_id) for model_id in sorted(models_with_ports)
  )
  nginx_bottom = "    }\n}\n"
  nginx_conf = nginx_top + nginx_blocks + nginx_bottom

  return compose_yaml, nginx_conf
