
fp = open('dump.txt')

head_vals = {}
for line in fp:
  es = iter(line.strip().split())
  head = float(next(es))

  vals = []
  for fact in es:
    index, weight = fact.split(':') 
    if int(index)%7 == 0:
      vals.append( 0.0 )
    else:
      vals.append( float(weight) )

  head_vals[head] = vals

for head, vals in sorted(head_vals.items(), key=lambda x:x[0]):
  
  print(head, ' '.join([f'{v}' for v in vals]))
