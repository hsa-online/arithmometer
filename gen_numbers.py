"""
Generates the "totals.csv" file to be used as input for trainer.py

"""

import random

import pandas as pd

VALUE_MIN = 0
VALUE_MAX = 16383
NUM_SAMPLES = 16384

data = []
pairs = set()

for i in range(NUM_SAMPLES):
    while True:
        num1 = random.randrange(VALUE_MIN, VALUE_MAX) / VALUE_MAX
        num2 = random.randrange(VALUE_MIN, VALUE_MAX) / VALUE_MAX
        pair = (min(num1, num2), max(num1, num2))
        if pair not in pairs:
            pairs.add(pair)
     
            total = num1 + num2
            row = [num1, num2, total]
            data.append(row)
            row = [num2, num1, total]
            data.append(row)

            break

df = pd.DataFrame(data, columns=['a', 'b', 'total'])

# Not actually necessary to convert when we work with floats, 
#   but we can try to train from integers ;-)
df['a'] = df['a'].astype(float, errors = "raise")
df['b'] = df['b'].astype(float, errors = "raise")
df['total'] = df['total'].astype(float, errors = "raise")

df.to_csv('totals.csv', index=False)
