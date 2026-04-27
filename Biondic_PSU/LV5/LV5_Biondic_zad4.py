"""
Što primjećujete kod vrednovanja ovog modela?

Primjećuje se da logistička regresija postiže vrlo visoku točnost, preciznost i odziv slično kao što su postizali KNN ili stabla 
odlučivanja u prethodnim zadacima. Matrica zabune pokazuje minimalan broj pogrešno klasificiranih uzoraka odnosno FP i FN su vrlo mali.

Što je uzrok dobivenim rezultatima?

Jedna od glavnih uzroka ovakvih rezultata je linearna odvojivost kod koje senzorski podaci za temperaturu i CO2 
u skupu podataka vrlo jasno razdvajaju klase. Primjer toga možemo vidjeti u prvom zadatku gdje s porastom temperature i CO2 raste 
i vjerojatnost da je prostorija zauzeta. Također je i logistička funkcija jedan od uzroka ovakvog rezultata jer je idealna za binarne 
klasifikacijske probleme gdje postoji jasna granica kao što imamo jasnu granicu u prvom zadatku između "Slobodne" i "Zauzete" prostorije. 
Još jedan od uzroka bila bi kvaliteta značajki. Primjer toga je što je Senzor CO2 vrlo precizan indikator prisutnosti ljudi u prostoriji 
te pomoću njega model može s velikom sigurnošću donijeti ispravnu odluku o klasi. 

"""