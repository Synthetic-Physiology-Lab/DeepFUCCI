import pandas as pd
import matplotlib.pyplot as plt
from fucciphase.plot import plot_raw_intensities

ref_data = pd.read_csv("ht1080_reference.csv")

plot_raw_intensities(ref_data, channel1="cyan", channel2="magenta", time_column="percentage", time_label="Cell cycle percentage", lw=4) 
plt.savefig("reference_ht1080.pdf")
plt.show()
