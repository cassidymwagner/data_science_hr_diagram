import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('planets_new.csv')
def add_columns(column_list):
    df_new = pd.read_csv('planets_new.csv')
    df_complete = pd.read_csv('planets_complete.csv', header=146)
    df_new.join(df_complete[column_list]).to_csv('planets_new.csv')
# add_columns(['st_glon', 'st_glat'])
# add_columns(['st_elon', 'st_elat'])
# add_columns(['st_lum'])


# plotting by year of discovery
# color setup
vcs = df.pl_discmethod.value_counts()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] # maybe color_cycle
discmethod_dict = dict([(vcs.index[i], colors[i]) for i in range(len(vcs))])
plt.show()

# loop through discovery year
from time import sleep
for year in df.pl_disc.drop_duplicates().sort_values():
    df_temp = df[df.pl_disc <= year]
    df_temp.plot.scatter('ra', 'dec', 
        xlim=[0, 360], ylim=[-90,90], 
        s=df_temp.pl_bmassj, 
        c=list(df_temp.pl_discmethod.map(discmethod_dict)),
        title=year)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in discmethod_dict.values()][:4]
    plt.legend(markers, list(discmethod_dict.keys())[:4], 
        numpoints=1, ncol=2, bbox_to_anchor=(0.82, -0.15))
    plt.show(block=False)
    sleep(0.3)
    list(discmethod_dict.keys())

for year in df.pl_disc.drop_duplicates().sort_values():
    df_temp = df[df.pl_disc <= year]
    df_temp.plot.scatter('pl_orbsmax', 'pl_bmassj', 
        xlim=[df.pl_orbsmax.min(), df.pl_orbsmax.max()], ylim=[df.pl_bmassj.min(),df.pl_bmassj.max()], 
        c=list(df_temp.pl_discmethod.map(discmethod_dict)),
        s=2, title=year, loglog=True); 
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in discmethod_dict.values()][:4]
    plt.legend(markers, list(discmethod_dict.keys())[:4], 
        numpoints=1, ncol=2, bbox_to_anchor=(0.82, -0.15))
    plt.legend
    plt.show(block=False)
    # sleep(0.3)
    

# galactic plane
df.plot.scatter('ra', 'dec', s=1)
df.plot.scatter('st_elon', 'st_elat', s=1)
plt.show()

# kepler lol
df.plot.hexbin('ra', 'dec')
plt.show()


# separation vs. planet mass
df.pl_orbsmax.plot.kde(xlim=[0,100])
df.pl_bmassj.plot.kde()
df[df.pl_orbsmax < 10].plot.scatter('pl_orbsmax', 'pl_bmassj', s=1)
plt.show()

# discovery method
df.pl_discmethod.value_counts()
df.pl_discmethod.value_counts().plot.bar()
plt.show()

df.pl_discmethod = pd.Categorical(df.pl_discmethod)
df['pl_discmethod_code'] = df.pl_discmethod.cat.codes
plt.show()

def compare():
    df[df.pl_discmethod=='Transit'].plot.scatter('ra', 'dec', s=1)
    df[df.pl_discmethod=='Radial Velocity'].plot.scatter('ra', 'dec', s=1)
    df[df.pl_discmethod=='Microlensing'].plot.scatter('ra', 'dec', s=1)
compare()
plt.show()

def compare(variable, kind):
    df[variable][df.pl_discmethod=='Transit'].plot(kind=kind)
    df[variable][df.pl_discmethod=='Radial Velocity'].plot(kind=kind, c='crimson')
    # df[variable][df.pl_discmethod=='Microlensing'].plot(kind=kind)
compare('pl_radj', 'kde')
plt.show()
compare('pl_orbsmax', 'kde')
plt.show()
compare('st_optmag', 'kde')
plt.show()

# exoplanet HR
df['st_log_teff'] = np.log10(df.st_teff)
df.plot.scatter('st_log_teff', 'st_lum', s=1)
plt.show()
