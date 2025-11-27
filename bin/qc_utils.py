import numpy as np
import matplotlib.pyplot as plt

def plot_ts_diagram(s,t,meta,directory, show=TRUE, save=True)
    #creates a TS plot, uses salinity and temp arrays, meta data for sites, serial, pressure, and year_n
    fig_ts, ax_ts = plt.subplots(figsize=(8, 6))
    sc = ax_ts.scatter(
        s, t,
        c=np.arange(len(t)), cmap="viridis",
        edgecolor='k', alpha=0.7
    )
    ax_ts.set_xlabel('Salinity')
    ax_ts.set_ylabel('Temperature')
    ax_ts.set_title('T-S Diagram')
    cbar = plt.colorbar(sc, label='Index')

    # read out site and sitename from yaml
    site = meta["site"]
    subsite = meta["subsite"]
    serial = meta["serial"]
    year_str = str(meta["year_n"])
    pres = meta["pres"]

    ts_plot_path = f'{directory}{site}{year_str}{subsite}{pres}_{serial}_TS_plot.png'
    plt.savefig(ts_plot_path, dpi=300, bbox_inches='tight')
    plt.show(block=True)