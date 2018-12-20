import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture, cluster, neighbors

stars = pd.read_csv('hygdata_v3.csv')

stars.loc[:,'temp'] = 4600 * (1 / (0.92 * stars.ci + 1.7) + 1 / (0.92 * stars.ci + 0.62))
stars.loc[:,'logtemp'] = np.log10(stars.temp)
stars.loc[:,'loglum'] = np.log10(stars.lum)

stars.plot.scatter(x='temp', y='lum', loglog=True, s=1)
plt.show()



fig, ax = plt.subplots()
stars.plot.scatter(x='temp', y='lum', s=1,
                loglog=True,
                c='logtemp', colormap='coolwarm_r',
                ax=ax)
ax.set_xlabel('Temperature (K)'); ax.set_ylabel(r'Luminosity ($L_\odot$)')
ax.invert_xaxis()
plt.show()

stars.plot.scatter(x='dist', y='lum', s=2, loglog=True, xlim=(1,1e6))
plt.show()

data = np.log10(stars.loc[:,['dist', 'lum']]).replace(
     [np.inf, -np.inf], np.nan).dropna()


close_stars = stars[stars.dist < 50]
fig, ax = plt.subplots()
close_stars.plot.scatter(x='temp', y='lum', s=1,
                loglog=True,
                c='logtemp', colormap='coolwarm_r',
                ax=ax)
ax.set_xlabel('Temperature (K)'); ax.set_ylabel(r'Luminosity ($L_\odot$)')
ax.invert_xaxis()
plt.show()

n_clusters = 3
bgmm = mixture.BayesianGaussianMixture(n_components=n_clusters, 
    weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=1e10)



#clustering_data = close_stars.loc[:, ['temp', 'lum']].dropna()
clustering_data = stars.loc[:, ['temp', 'lum']].dropna().sample(10000)

bgmm.fit(np.log10(clustering_data))
labels = bgmm.predict(np.log10(clustering_data))

fig, ax = plt.subplots()
clustering_data.plot.scatter(x='temp', y='lum', s=3,
                loglog=True, colormap='tab10',
                c=labels, colorbar=False,
                ax=ax)
ax.set_xlabel('Temperature (K)'); ax.set_ylabel(r'Luminosity ($L_\odot$)')
ax.invert_xaxis()
plt.show()
