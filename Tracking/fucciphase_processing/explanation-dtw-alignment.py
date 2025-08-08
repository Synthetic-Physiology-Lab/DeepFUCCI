import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fucciphase.phase import (
    estimate_percentage_by_subsequence_alignment,
)
from fucciphase.plot import plot_query_vs_reference_in_time
import scipy.interpolate as interpolate


# Read the reference curve
reference_file = "fuccisa_ht1080_reference.csv"
reference_df = pd.read_csv(reference_file)
# add a fake TRACK_ID
reference_df["TRACK_ID"] = 1
timestep = reference_df["time"][1] - reference_df["time"][0]
cyan_channel = "cyan"
magenta_channel = "magenta"

# Create a test curve
# * take selected datapoints
# * scale the intensity or introduce an offset

start_idx = 0
end_idx = 40
df = reference_df.iloc[start_idx:end_idx]
df["cyan"] = df.loc[:, "cyan"] + 0.6
df["magenta"] = df.loc[:, "magenta"] + 0.7

estimate_percentage_by_subsequence_alignment(
    df,
    dt=timestep,
    channels=[cyan_channel, magenta_channel],
    reference_data=reference_df,
    track_id_name="TRACK_ID",
)

plot_query_vs_reference_in_time(
    reference_df,
    df,
    channels=["cyan", "magenta"],
    channel_titles=["mTurquoise2", "miRFP670"],
    fig_title="Test",
    lw=3,
)
plt.suptitle("Test")
plt.tight_layout()

plt.savefig("explain_dtw_query.pdf")
plt.savefig("explain_dtw_query.svg")
plt.show()


g1_cyan_interpolate = interpolate.interp1d(
    df["percentage"].iloc[:31], df["cyan"].iloc[:31]
)
g1_magenta_interpolate = interpolate.interp1d(
    df["percentage"].iloc[:31], df["magenta"].iloc[:31]
)

# scale G1 phase artificially
n_samples = 30
scaling = 2
new_perc_g1 = np.linspace(0, n_samples, num=(n_samples * scaling) + 1)
dt = df["time"].iloc[1] - df["time"].iloc[0]
new_time = np.linspace(
    start=0, stop=df["time"].iloc[31] * scaling, num=(n_samples * scaling) + 1
)

cyan_new = np.append(g1_cyan_interpolate(new_perc_g1), df["cyan"].iloc[31:].to_numpy())
combined_time = np.append(
    new_time, new_time[-1] + df["time"].iloc[31:].to_numpy() - df["time"].iloc[31]
)
magenta_new = np.append(
    g1_magenta_interpolate(new_perc_g1), df["magenta"].iloc[31:].to_numpy()
)

new_df = pd.DataFrame({"time": combined_time, "cyan": cyan_new, "magenta": magenta_new})
new_df["TRACK_ID"] = 1
estimate_percentage_by_subsequence_alignment(
    new_df,
    dt=timestep,
    channels=[cyan_channel, magenta_channel],
    reference_data=reference_df,
    track_id_name="TRACK_ID",
)


plot_query_vs_reference_in_time(
    reference_df,
    new_df,
    channels=["cyan", "magenta"],
    channel_titles=["mTurquoise2", "miRFP670"],
    fig_title="Test",
    lw=3,
)
plt.savefig("explain_dtw_query_distorted.pdf")
plt.savefig("explain_dtw_query_distorted.svg")
plt.show()
