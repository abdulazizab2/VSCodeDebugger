"""
    This tutorial will illustrate what is a condintional debugger and why it is useful
    1- When to use a conditional debugger
    2- Expression and hit count
"""
# step1: Execute the code
# Question: If I start debugging, where will the debugger stop? ()
# step2: Add a condintional breakpoint.
            # Question: Where should the condintional breakpoint be added?



import numpy as np

x = 0
for i in range(1000):
    random_value = np.random.randint(low=0, high=100)
    x += np.log(random_value)

    
print(f'x: {x}, random_value: {random_value}')
    
