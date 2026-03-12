brojevi = []

while True:
    print('Unosi broj ili Done za kraj ')
    x= input()

    if x == "Done":
        break
    
    if x.isdigit():
        broj = float(x)
        brojevi.append(broj)
    else:
        print("Pogrešan unos! Molimo unesite broj.")
if len(brojevi) > 0:
    print("Broj unesenih brojeva:", len(brojevi))
    print("Srednja vrijednost:", sum(brojevi) / len(brojevi))
    print("Minimalna vrijednost:", min(brojevi))
    print("Maksimalna vrijednost:", max(brojevi))

    brojevi.sort()
    print("Sortirana lista:", brojevi)
else:
    print("Nije unesen nijedan broj.")


