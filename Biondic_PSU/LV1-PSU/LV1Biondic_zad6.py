ime = "SMSSpamCollection.txt"

dat = open(ime)

ham_broj = 0
spam_broj = 0

ham_rijeci = 0
spam_rijeci = 0

spam_usklicnik = 0

for linija in dat:
    linija = linija.rstrip()
    rijeci = linija.split()
    
    oznaka = rijeci[0]

    broj = len(rijeci) - 1
    
    if oznaka == "ham":
        ham_broj = ham_broj + 1
        ham_rijeci = ham_rijeci + broj
    
    if oznaka == "spam":
        spam_broj = spam_broj + 1
        spam_rijeci = spam_rijeci + broj
        
        if linija.endswith("!"):
            spam_usklicnik = spam_usklicnik + 1


prosjek_ham = ham_rijeci / ham_broj
prosjek_spam = spam_rijeci / spam_broj

print("Prosjecan broj rijeci (ham):", prosjek_ham)
print("Prosjecan broj rijeci (spam):", prosjek_spam)
print("Spam poruke koje zavrsvaju usklicnikom:", spam_usklicnik)