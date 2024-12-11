from random import randint

mass = 8810324116.227

f = open("64k_body.txt", "w")
f.write("64000\n")
for i in range(40):
    for j in range(40):
        for k in range(40):
            f.write(f"{mass} {i} {j} {k} {randint(1, 10000)} {randint(1, 10000)} {randint(1, 10000)}\n")

f.close()