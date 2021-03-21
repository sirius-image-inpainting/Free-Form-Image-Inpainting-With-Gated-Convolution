import numpy as np
import model.data
import matplotlib.pyplot as plt

#  dataset = model.data.PlacesDataset('data/train/', transform=False)
#  image = dataset[0]
#  print(image)

mask = model.data.generate_random_mask()

plt.imshow(mask)
plt.show()
