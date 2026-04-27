import numpy as np
import matplotlib.pyplot as plt

def sahovnica(velicina_kvadrata, broj_po_visini, broj_po_sirini):
    
    crni = np.zeros((velicina_kvadrata, velicina_kvadrata))
    bijeli = np.ones((velicina_kvadrata, velicina_kvadrata)) * 255

    redovi = []

    for i in range(broj_po_visini):
        stupci = []

        for j in range(broj_po_sirini):
            if (i + j) % 2 == 0:
                stupci.append(crni)
            else:
                stupci.append(bijeli)

        red = np.hstack(stupci)
        redovi.append(red)

    slika = np.vstack(redovi)

    return slika


img = sahovnica(50, 4, 5)

plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title("Šahovnica")
plt.show()