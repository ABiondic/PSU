
print('Unesi radne sate: ')
x=float(input())
print('Unesi cijenu radnog sata: ')
y=float(input())

def total_euro(x, y):
    ukupno = x * y
    return ukupno

zarada = total_euro(x,y)
print('Radni sati: ',x, "h")
print('eura/h:',y)
print('Ukupno: ',zarada, "eura")