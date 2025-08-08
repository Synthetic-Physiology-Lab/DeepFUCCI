import pandas as pd
import matplotlib.pyplot as plt

ref_data = pd.read_csv("fuccisa_ht1080_reference.csv")

channel1="cyan"
channel2="magenta"
time_column="time"
percentage_column="percentage"
time_label="Time / h"
percentage_label="Cell cycle percentage"
color1 = "cyan"
color2 = "magenta"


ch1_intensity = ref_data[channel1]
ch2_intensity = ref_data[channel2]
t = ref_data[time_column]
perc = ref_data[percentage_column]

fig, ax1 = plt.subplots()

# prepare axes
ax1.set_xlabel(time_label)
ax1.set_ylabel("mTurquoise2")
ax1.tick_params(axis="y", labelcolor=color1)
ax2 = ax1.twinx()
ax2.set_ylabel("miRFP670")
ax2.tick_params(axis="y", labelcolor=color2)

ax3 = ax1.twiny()

# plot signal
ax1.plot(t, ch1_intensity, color=color1, lw=4)
ax2.plot(t, ch2_intensity, color=color2, lw=4)
ax3.plot(perc, ch1_intensity, color=color1, lw=4)
ax3.set_xlabel(percentage_label)

fig.tight_layout()
plt.savefig("reference_ht1080.pdf")
plt.show()
