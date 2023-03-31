import deeplake

ds = deeplake.load('hub://activeloop/wiki-art')
dataloader = ds.pytorch(num_workers=0, batch_size=4, shuffle=False)

dataloader = ds.tensorflow()
