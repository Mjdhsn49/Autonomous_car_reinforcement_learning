import random
from itertools import count
#import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig,axs = plt.subplots(2)

def plot_overtime(x,y,y2):
	
	fig.suptitle('Reward and Epsilon over episodes')
	axs[0].plot(x,y, 'tab:orange', label='Reward')
	axs[1].plot(x,y2,'tab:green',label='Epsilon')
	axs[0].legend(loc='upper right')
	axs[1].legend(loc='upper right')
	plt.show()
