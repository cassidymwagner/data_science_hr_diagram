import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


df = pd.read_csv('planets_new.csv')
#def add_columns(column_list):
#    df_new = pd.read_csv('planets_new.csv')
#    df_complete = pd.read_csv('planets_complete.csv', header=146)
#    df_new.join(df_complete[column_list]).to_csv('planets_new.csv')
#add_columns(['st_glon', 'st_glat'])
#add_columns(['st_elon', 'st_elat'])
#add_columns(['st_lum'])


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
