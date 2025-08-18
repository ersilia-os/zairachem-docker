import csv, json, time, requests
from zairachem.base.utils.logging import logger
import numpy as np

DEFAULT_BATCH_SIZE = 100_000


class BinaryStreamClient:
  def __init__(self, csv_path, url=None, batch_size=DEFAULT_BATCH_SIZE):
    self.logger = logger
    self.batch_size = batch_size
    self.csv_path = csv_path
    self.url = url
    self.input_data = self._load_data()

  def _load_data(self) -> list:
    with open(self.csv_path, "r") as f:
      reader = csv.reader(f)
      next(reader)
      return [row[0] for row in reader]

  def decode_binary_stream(self, response, chunk_size=8192):
    it = response.iter_content(chunk_size=chunk_size)
    header_buf = bytearray()
    for chunk in it:
      header_buf.extend(chunk)
      if b"\n" in header_buf:
        header_line, remainder = header_buf.split(b"\n", 1)
        break
    else:
      raise IOError("Stream ended before header")
    meta = json.loads(header_line.decode("utf-8"))
    dtype = np.dtype(meta["dtype"])
    shape = tuple(meta["shape"])
    total_bytes = int(np.prod(shape) * dtype.itemsize)
    arr = np.empty(shape, dtype=dtype)
    view = memoryview(arr).cast("B")
    read = len(remainder)
    view[:read] = remainder

    for chunk in it:
      view[read : read + len(chunk)] = chunk
      read += len(chunk)
      if read >= total_bytes:
        break

    if read < total_bytes:
      raise IOError(f"Incomplete read: got {read} of {total_bytes} bytes")
    return arr, meta

  def run(self):
    total_time = 0.0
    arrays, results = [], None
    try:
      for i in range(0, len(self.input_data), self.batch_size):
        batch = self.input_data[i : i + self.batch_size]
        params = {"output_type": "heavy"}

        self.logger.debug(f"Sending batch {i} to the api")
        start_send = time.perf_counter()
        try:
          response = requests.post(self.url, json=batch, params=params, stream=True)
          response.raise_for_status()
        except requests.RequestException as e:
          self.logger.error(f"✖ Network error on batch {i}: {e}")
          break
        end_send = time.perf_counter()

        send_time = end_send - start_send
        total_time += send_time
        self.logger.info(f"Request sent in {send_time:.4f}s to the api")

        start_recv = time.perf_counter()
        try:
          array, results = self.decode_binary_stream(response)
        except IOError as e:
          self.logger.error(f"✖ Decode error on batch {i}: {e}")
          break
        end_recv = time.perf_counter()

        recv_time = end_recv - start_recv
        total_time += recv_time

        arrays.extend(array)
    finally:
      self.logger.info(f"Results are fetched in total elapsed time: {total_time:.4f}s")

    if results and arrays:
      stacked = np.vstack(arrays)
      results.update({
        "shape": stacked.shape,
        "data": stacked,
        "inputs": self.input_data,
      })
    return results
