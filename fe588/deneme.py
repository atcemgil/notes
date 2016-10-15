l = -1
r = 1
delta = 0.1
steps = (r-l)/delta+1

print '-'*20
print('| '),
print('x'),
print('| '),
print('3*x**2 + 2*x + 3'),
print('| ')

for i in range(0,int(steps)):
    x = l+i*delta
    print '-'*20
    print('| '),
    print(x),
    print('| '),
    print(3*x**2 + 2*x + 3),
    print('| ')
    