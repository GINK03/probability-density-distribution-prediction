import numpy as np
import json
import pickle
import sys

if '--invert' in sys.argv:
  fp = open('./dump.txt')
  # index -> Y
  # weigts -> X
  heads = set()

  for line in fp:
    ents = iter(line.strip().split())
    head = float(next(ents))
    heads.add(head)

  head_index = {}
  for index, head in enumerate(sorted(list(heads), key=lambda x:float(x))):
    print(index, head)
    head_index[head] = index

  fp = open('./dump.txt')
  day_index_weight = {}
  for line in fp:
    ents = iter(line.strip().split())
    head = float(next(ents))
    for val in ents:
      day, weight = val.split(':')
      day, weight = int(day), float(weight)

      if day_index_weight.get(day) is None:
        day_index_weight[day] = {}

      index = head_index[head]
      day_index_weight[day][index] = weight
  open('day_index_weight.pkl', 'wb').write( pickle.dumps(day_index_weight) )

if '--np' in sys.argv:
  day_index_weight = pickle.loads( open('day_index_weight.pkl', 'rb').read())

  Xs, Ys, Xst, Yst = [], [], [], []

  for day, index_weight in day_index_weight.items():
    if day%7 != 0:
      print(day)
      ys = []
      for index, weight in sorted(index_weight.items(), key=lambda x:x[0]):
        ys.append( weight ) 
      Ys.append(ys)
      Xs.append(day /25000)
    else:
      yst = []
      for index, weight in sorted(index_weight.items(), key=lambda x:x[0]):
        yst.append( weight ) 
      Yst.append(yst)
      Xst.append(day /25000 )

  Xs, Ys, Xst, Yst = map(lambda x:np.array(x, dtype=float), [Xs, Ys, Xst, Yst])
  print(Ys.shape)

  open('dataset.pkl', 'wb').write( pickle.dumps( (Xs, Ys, Xst, Yst) ) )
