from zairachem.describe.descriptors.api import BinaryStreamClient
from zairachem.describe.descriptors.utils import Hdf5DataLoader

# api = BinaryStreamClient(csv_path="example_tiny.csv", url="http://localhost:8002/run")

loader = Hdf5DataLoader()
loader.load(
  "/home/abellegese/Desktop/projects/zairachem-docker/model_dir/descriptors/eos5axz/raw.h5"
)
print(loader.values)

# res = api.run()
# print(res["dtype"])
# print(res["shape"])
# print(res["dims"][:3])
# print(res["data"][:3])
# # print(res["inputs"][:3])
