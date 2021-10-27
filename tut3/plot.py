import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

d = np.loadtxt('t1_cpu.txt')

plt.imshow(d)
plt.savefig('t1_cpu.png')
plt.close()

d = np.loadtxt('tfinal_cpu.txt')

plt.imshow(d)
plt.savefig('tfinal_cpu.png')
plt.close()

d = np.loadtxt('tfinal_gpu.txt')

plt.imshow(d)
plt.savefig('tfinal_gpu.png')
plt.close()



