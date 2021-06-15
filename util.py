from matplotlib import pyplot as plt
x  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
y1 = [0.0001, 0.3652, 0.723, 0.9944, 0.9994, 0.9998, 0.9999, 0.9999, 0.9999, 1.0, 1.0]
y3 = [0.0001, 0.3626, 0.7187, 0.9836, 0.9994, 0.9998, 0.9999, 0.9999, 0.9999, 1.0, 1.0]
y0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1908, 0.4595, 0.7303, 1.0]
y2 = [0.0001, 0.3659, 0.7258, 0.999, 0.9996, 0.9998, 0.9999, 1.0, 1.0, 1.0, 1.0]

plt.figure()
plt.plot(x,y1, label='Label Bias')
plt.plot(x,y3, label='FAIR')
plt.plot(x,y0, label='Unconstrained', color='y')
plt.plot(x,y2, label='Ideal')
plt.legend()
plt.xlabel('Fraction of training data checked')
plt.ylabel('Fraction of mislabel identified')
plt.savefig('figs/TracIn_comparison.png')
plt.show()