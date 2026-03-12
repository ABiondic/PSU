print('Unesi broj od 0 do 1 ')
x=float(input())

if x < 0.0 or x > 1.0:
    print('Greska: pogresan unos, ocjena mora biti izmedu 0.0 i 1.0')
elif x >= 0.9:
    print('A')
elif x >= 0.8:
    print('B')
elif x >= 0.7:
    print('C')
elif x >= 0.6:
    print('D')
else:
    print('F')


