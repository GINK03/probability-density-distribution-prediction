import numpy as np

SIZE = 5000
obs_freq = {}
for i in range(SIZE):
  sample1 = np.random.normal(loc=-2*np.sin(i/10), scale=1, size=10000)
  sample2 = np.random.normal(loc=5*np.cos(i/10), scale=2, size=30000)

  sample = np.append(sample1, sample2, axis=0)
  for s in sample.tolist():
    #print(s, s//0.001000)
    obs = s//0.5000
    if obs_freq.get(obs) is None:
      obs_freq[obs] = [0]*SIZE
    obs_freq[obs][i] += 1

vs = []
for inv in list(map(list, zip(*list([val for obs, val in sorted(obs_freq.items(), key=lambda x:x[0])])))):
  vs.append( max(inv) )
for obs, freq in sorted(obs_freq.items(), key=lambda x:x[0]):
  print(obs, ' '.join(map(str, [ f/vs[index] for index, f in enumerate(freq)])))
