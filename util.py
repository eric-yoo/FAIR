from matplotlib import pyplot as plt
x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
y0 = [0.0001, 0.5422, 0.974, 0.9997, 0.9999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
y1 = [0.0001, 0.5363, 0.9691, 0.9994, 0.9994, 0.9996, 0.9998, 1.0, 1.0, 1.0, 1.0]
y2 = [0.0001, 0.3269, 0.6921, 0.8936, 0.9097, 0.9207, 0.93, 0.9437, 0.9611, 0.9808, 1.0]
plt.figure()
plt.plot(x,y0, label='TracIn on Unbiased Model')
plt.plot(x,y1, label='TracIn on Reweighted Model')
plt.plot(x,y2, label='TracIn on Biased Model')
plt.legend()
plt.savefig('figs/TracIn_comparison.png')
plt.show()