score = {"john": ('Physics', 80, 'Science', 95),"Daniel": ('History', 75, 'Science', 90), "Mark": ('Maths', 100,'Social', 95)}
for key, value in score.items():
    print(key, ":", [(value)])

x = 'pwwkew'

sub = x[1::2]

long, length = sub, 1

for i in x[1:]:
    if ord(sub[-1]) <= ord(i):
        sub += i
        if len(sub) > length:
            length = len(sub)
            long = sub
    else:
        sub = i
print(long)








