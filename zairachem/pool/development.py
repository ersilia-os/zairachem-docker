from pool.bagger.bagger import XGetter

g = XGetter(path=None)
df = g.get()
print(df.head())
