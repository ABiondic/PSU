ime = "song.txt"

dat = open(ime)

rjecnik = {}

for linija in dat:
    linija = linija.strip()
    rijeci = linija.split()
    
    for rijec in rijeci:
        if rijec in rjecnik:
            rjecnik[rijec] += 1
        else:
            rjecnik[rijec] = 1

jednom = []

for rijec in rjecnik:
    if rjecnik[rijec] == 1:
        jednom.append(rijec)

print("Broj rijeci koje se pojavljuju samo jednom:", len(jednom))
print("Te rijeci su:")
for r in jednom:
    print(r)