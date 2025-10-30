import csv, json, hashlib, time, requests, re
import pandas as pd
from zairachem.base.utils.logging import logger
from isaura.manage import IsauraReader, IsauraWriter
import numpy as np

_RING_TOKEN_RE = re.compile(r"%(?:\d{2})|[0-9]")
_ALLOWED_RE = re.compile(
  r"^[A-Za-z0-9@+\-\[\]\(\)=#$:.\\/%,*\.]+(?:\s+[A-Za-z0-9@+\-\[\]\(\)=#$:.\\/%,*\.]+)?$"
)
DEFAULT_BATCH_SIZE = 100_000


class BinaryStreamClient:
  def __init__(self, csv_path, model_id=None, url=None, project_name=None, batch_size=DEFAULT_BATCH_SIZE):
    self.logger = logger
    self.batch_size = batch_size
    self.csv_path = csv_path
    self.url = url
    self.input_data, self.input_header = self._load_data()
    self._feature_len = None
    self.project_name = project_name
    self.model_id = model_id

  def _load_data(self) -> list:
    with open(self.csv_path, "r") as f:
      reader = csv.reader(f)
      h = next(reader)
      return ([row[0] for row in reader], h)

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

  def _strip_brackets(self, s):
    return re.sub(r"\[[^\[\]]*\]", "", s)

  def _balanced(self, s, open_ch, close_ch):
    c = 0
    for ch in s:
      if ch == open_ch:
        c += 1
      elif ch == close_ch:
        c -= 1
        if c < 0:
          return False
    return c == 0

  def _rings_even(self, s):
    t = self._strip_brackets(s)
    tokens = _RING_TOKEN_RE.findall(t)
    counts = {}
    for tok in tokens:
      counts[tok] = counts.get(tok, 0) + 1
    return all(v % 2 == 0 for v in counts.values())

  def _is_smiles(self, text):
    if not isinstance(text, str):
      return False
    s = text.strip()
    if not s or "\n" in s or "\t" in s or "  " in s:
      return False
    if not _ALLOWED_RE.fullmatch(s):
      return False
    if not self._balanced(s, "(", ")"):
      return False
    if not self._balanced(s, "[", "]"):
      return False
    if "()" in s or "[]]" in s or "[[" in s:
      return False
    if not self._rings_even(s):
      return False
    return True

  def _placeholder_row(self):
    if self._feature_len is None:
      raise RuntimeError("Feature length unknown")
    return np.full((1, self._feature_len), np.nan, dtype=float)

  def _try_request(self, batch):
    params = {"output_type": "heavy"}
    response = requests.post(self.url, json=batch, params=params, stream=True)
    response.raise_for_status()
    array, results = self.decode_binary_stream(response)
    if self._feature_len is None:
      arr0 = array[0] if isinstance(array, (list, tuple)) else array
      arr0 = np.asarray(arr0)
      self._feature_len = arr0.shape[-1] if arr0.ndim > 1 else arr0.shape[0]
    return array, results

  def _fetch_or_split(self, batch, idx_offset, depth=0):
    try:
      array, results = self._try_request(batch)
      if isinstance(array, np.ndarray):
        arrays_out = [array[i : i + 1] for i in range(array.shape[0])]
      else:
        arrays_out = [np.asarray(row).reshape(1, -1) for row in array]
      return arrays_out, [True] * len(arrays_out), results
    except (requests.RequestException, IOError):
      if len(batch) == 1:
        return [None], [False], None
      mid = len(batch) // 2
      left_arrays, left_mask, left_res = self._fetch_or_split(batch[:mid], idx_offset, depth + 1)
      right_arrays, right_mask, right_res = self._fetch_or_split(
        batch[mid:], idx_offset + mid, depth + 1
      )
      return left_arrays + right_arrays, left_mask + right_mask, right_res or left_res

  def run(self):
    try:
      df = pd.DataFrame(columns=["input"], data=self.input_data)
      r = IsauraReader(
        model_id=self.model_id,
        model_version="v1",
        input_csv=None,
        approximate=False,
        bucket=self.project_name,
      )
      df = r.read(df=df)
      values = df[df.columns.difference(["key", "input"])].values
      values = values.astype("float")
      cols = df.columns.difference(["key", "input"]).tolist()
      any_results = {
        "shape": values.shape,
        "dtype": "float",
        "data": values,
        "dims": cols,
        "inputs": self.input_data,
      }
      return any_results
    except SystemExit as e:
      logger.info("Fall back to the api for calculating the descriptors!")
      return self._run()

  def _get_ersilia_df(self, res):
     keys = [hashlib.md5(inp.encode()).hexdigest() for inp in res["inputs"]]
     return pd.DataFrame({"key": keys, "input": res["inputs"]}).join(
      pd.DataFrame(res["data"], columns=res["dims"], dtype=str)
    )

  def _run(self):
    total_time = 0.0
    n = len(self.input_data)
    per_item_rows = [None] * n
    valid_mask_input = [False] * n
    good_idx = []
    checked_input = []
    for i, s in enumerate(self.input_data):
      ok = self._is_smiles(s)
      valid_mask_input[i] = ok
      if ok:
        good_idx.append(i)
        checked_input.append(s)

    any_results = None
    try:
      for start in range(0, len(checked_input), self.batch_size):
        batch = checked_input[start : start + self.batch_size]
        batch_abs_offset = good_idx[start]
        t0 = time.perf_counter()

        arrays_out, _, results = self._fetch_or_split(batch, idx_offset=batch_abs_offset)
        total_time += time.perf_counter() - t0
        if results is not None:
          any_results = results
        for j, row in enumerate(arrays_out):
          per_item_rows[good_idx[start + j]] = row
    finally:
      self.logger.info(f"Total elapsed: {total_time:.4f}s")

    if self._feature_len is None:
      return None

    filled = []
    for row in per_item_rows:
      if isinstance(row, np.ndarray):
        filled.append(row)
      else:
        filled.append(self._placeholder_row())
    stacked = np.vstack(filled)

    if any_results is None:
      any_results = {}
    any_results.update({
      "shape": stacked.shape,
      "data": stacked,
      "inputs": self.input_data,
    })
    df = self._get_ersilia_df(any_results)
    try:
      w = IsauraWriter(input_csv=None, model_id=self.model_id, model_version="v1", bucket=self.project_name)
      w.write(df=df)
      time.sleep(10)
    except Exception as e:
      logger.error(e)
    return any_results
