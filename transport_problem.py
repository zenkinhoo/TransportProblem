# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 12:56:34 2022

@author: Lenovo T450
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 23:36:09 2022

@author: Lenovo T450
"""
import numpy as np

#kod ovog metoad se ne uzimaju u obzir cene
def severozapadni_ugao(ponuda, potraznja):
  #potrebno je napraviti kopije zato sto ce se vrednosti menjati kroz iteracije
    ponuda_copy = ponuda.copy()
    potraznja_copy = potraznja.copy()
    i = 0
    j = 0
    resenje = []
    while len(resenje) < len(ponuda) + len(potraznja) - 1:
        pon = ponuda_copy[i]
        pot = potraznja_copy[j]
        kolicina = min(pon, pot)
        #oduzimamu odredjenu kolicinu od kapaciteta i-tog skaldista i kapaciteta j-te destinacije
        ponuda_copy[i] -= kolicina
        potraznja_copy[j] -= kolicina

        resenje.append(((i, j), kolicina))
        #provera da li je sve potroseno sto je bilo u ponudi odredjenog skladista
        if ponuda_copy[i] == 0 and i < len(ponuda) - 1:
            i += 1
        #provera da li je sve potroseno sto je bilo u potraznji odredjene destinacije
        elif potraznja_copy[j] == 0 and j < len(potraznja) - 1:
          #ako jeste predji na sledecu
            j += 1
    return resenje

from copy import copy, deepcopy
def najmanje_cene(ponuda, potraznja,cene):
  #potrebno je napraviti kopije zato sto ce se vrednosti menjati kroz iteracije
    ponuda_copy = ponuda.copy()
    potraznja_copy = potraznja.copy()
    cene_copy=deepcopy(cene)
    trenutno_i=0
    trenutno_j=0
    resenje = []
    while len(resenje) < len(ponuda) + len(potraznja) - 1:#broj popunjenih polja mora biti m+n-1
        #prvo trazim najmanju cenu
        najmanja_cena=999
        for i in range(len(ponuda)):
          for j in range(len(potraznja)):
            if(cene_copy[i][j]<najmanja_cena):
              najmanja_cena=cene_copy[i][j]
              trenutno_i=i
              trenutno_j=j
        cene_copy[trenutno_i][trenutno_j]=9999
        #zatim provlacim kroz tu granu najvise sto moze
        pon = ponuda_copy[trenutno_i]
        pot = potraznja_copy[trenutno_j]
        kolicina = min(pon, pot)
        #onda oduzim te vrednosti
        ponuda_copy[trenutno_i] -= kolicina
        potraznja_copy[trenutno_j] -= kolicina
        if ponuda_copy[trenutno_i] == 0:
            for x in range(len(potraznja)):
              cene_copy[trenutno_i][x]=9999
        if potraznja_copy[trenutno_j] == 0:
            for x in range(len(ponuda)):
              cene_copy[x][trenutno_j]=9999
        resenje.append(((trenutno_i, trenutno_j), kolicina))
        
        
    return resenje;






def toAllocate(table2,s,d,supply,demand,cost):
	global sTot,dTot,iters
	if sTot==0 and dTot==0:
		return table2
	sortedRow=sortCost(s,d,0,cost)	
	sortedCol=sortCost(d,s,1,cost)	
	pRow=penaltyVal(sortedRow)	
	pCol=penaltyVal(sortedCol)	
	sortedAll=sortAll(pRow,pCol)	
	indexRow=indexCell(s,d,0,cost)		
	indexCol=indexCell(d,s,1,cost)		
	direc=sortedAll[0][3]
	if direc==0:
		idx=sortedAll[0][2]
		x=indexRow[idx][0][1]
		y=indexRow[idx][0][2]
	elif direc==1:
		idx=sortedAll[0][2]
		x=indexCol[idx][0][1]
		y=indexCol[idx][0][2]
	if supply[x]<demand[y]:
		table2[x][y]=supply[x]
		demand[y]-=supply[x]
		supply[x]=0
		sTot=sum(supply)
		dTot=sum(demand)
		for a in range(d):
			cost[x][a]=1000000
		toAllocate(table2,s,d,supply,demand,cost)
	elif supply[x]>demand[y]:
		table2[x][y]=demand[y]
		supply[x]-=demand[y]
		demand[y]=0
		sTot=sum(supply)
		dTot=sum(demand)
		for z in range(s):
			cost[z][y]=1000000
		toAllocate(table2,s,d,supply,demand,cost)
	elif supply[x]==demand[y]:
		table2[x][y]=supply[x]
		supply[x]=0
		demand[y]=0
		sTot=sum(supply)
		dTot=sum(demand)
		for a in range(d):
			cost[x][a]=1000000
		for z in range(s):
			cost[z][y]=1000000
		toAllocate(table2,s,d,supply,demand,cost)

def sortCost(d,s,flag,cost):
	box=[]
	for x in range(d):
		temp=[]
		for y in range(s):
			if flag==1:
				temp.append(cost[y][x])
			else:
				temp.append(cost[x][y])
		temp2=sorted(temp)
		box.append(temp2)
	return box

def penaltyVal(lists):
	penBox=[]
	x=0
	for row in lists:
		if(len(row)==1):
			penalty=row[0]
		else:
			penalty=row[1]-row[0]
		temp=[]
		temp.insert(0,penalty)
		temp.insert(1,row[0])
		temp.insert(2,x)
		penBox.append(temp)
		x+=1
	return penBox

def indexCell(d,s,flag,cost):
	box=[]
	for x in range(d):
		temp2=[]
		for y in range(s):
			temp=[]
			if flag==1:
				temp.insert(0,cost[y][x])
				temp.insert(1,y)
				temp.insert(2,x)
			else:
				temp.insert(0,cost[x][y])
				temp.insert(1,x)
				temp.insert(2,y)
			temp2.append(temp)
		temp2=sorted(temp2)
		box.append(temp2)
	return box

def sortAll(row,col):
	box=[]
	for x in row:
		x+=[0]
		box.append(x)
	for x in col:
		x+=[1]
		box.append(x)
	box.sort(key=lambda z: (-z[0], z[1]))
	return box

def vogel(s,d,supply,demand,cost):
	table=[]
	for x in range(s):
		temp=[]
		for y in range(d):
			temp.append(0)
		table.append(temp)
	toAllocate(table,s,d,supply,demand,cost)     
	return table

def transformResult(resultTable):
  res=[]
  for i in range(len(resultTable)):
      for j in range(len(resultTable[i])):
          if(resultTable[i][j]!=0):
              res.append(((i,j),resultTable[i][j]))
  return res

def nadjiVelicinu(res):
    lista_vrsta =[]
    lista_kolona = []
    for index,el in enumerate(res):
        vrsta,kolona=el[0]
        lista_vrsta.append(vrsta)
        lista_kolona.append(kolona)
    lista_vrsta.sort()
    lista_kolona.sort()
    return lista_vrsta[-1],lista_kolona[-1]

def dopuniNulama(res,br_vrsta,br_kolona):
    broj_el=0
    m,n = nadjiVelicinu(res)
    matrica = np.full((m+1, n+1), -1)

    
    for index,el in enumerate(res):
        i,j=el[0]
        matrica[i][j]=el[1]
    
    
    for index,el in enumerate(res):
        broj_el+=1
    if  broj_el<br_vrsta+br_kolona-1:
        razlika= br_vrsta+br_kolona-1-broj_el
        for i in range(m):
            if razlika==0:
                break
            for i in range(n):
                if matrica[i][j]==-1:
                    matrica[i][j]=0
                    razlika-=1
                    if razlika==0:
                        break
                        
    result=[]
    for i in range(len(matrica)):
        for j in range(len(matrica[i])):
            if(matrica[i][j]!=-1):
                result.append(((i,j),matrica[i][j]))    
    return result

def izbalansiraj_problem(ponuda, potraznja, cene, penali=None):
    suma_ponuda = sum(ponuda)
    suma_potraznja = sum(potraznja)
    
    if suma_ponuda < suma_potraznja:#ako je ponuda manja od potraznje potrebno je napraviti jos jedno skladiste
        #if nove_cene is None:
        #    raise Exception('potrebno je dodati nove cene zbog finansijskog gubitka zbog nezadovoljene potraznje')
        nova_ponuda = ponuda + [suma_potraznja-suma_ponuda]
        nove_cene = cene + [penali]#posto smo dodali novu vrstu u matrici cena potrebno je ispuniti matricu sa novim cenama
        return nova_ponuda, potraznja, nove_cene
    if suma_ponuda > suma_potraznja:#potrebno je napraviti novu destinaciju
        nova_potraznja = potraznja + [suma_ponuda - suma_potraznja]
        #nove_cene = cene + [[0 for _ in potraznja]]#sve nove cene postavi na nula, za neiskorisceni kapacitet nema troskova
        for i in range(len(ponuda)):
            cene[i].append(0)
        
        return ponuda, nova_potraznja, cene
    return ponuda, potraznja, cene

def izracunaj_potencijale(pocetno_resenje, cene):
    

    u = [None] * len(cene)
    v = [None] * len(cene[0])
    u[0] = 0 #pocetni potencija u=0
    pocetno_resenje_copy = pocetno_resenje.copy()
    while len(pocetno_resenje_copy) > 0:
        for index, bv in enumerate(pocetno_resenje_copy):
            i, j = bv[0]#vrsta i kolona kroz koju je prosla odredjena kolicina robe
            if u[i] is None and v[j] is None: continue   
            cena = cene[i][j]
            if u[i] is None:
                u[i] = cena - v[j]
            else: 
                v[j] = cena - u[i]
            pocetno_resenje_copy.pop(index)
            break
            
    return u, v 

def izracunaj_nove_cene(pocetno_resenje, cene, u, v):
    nove_cene = []
    for i, vrsta in enumerate(cene):
        for j, cena in enumerate(vrsta):
            nebazna = all([polje[0] != i or polje[1] != j for polje, vrednost in pocetno_resenje])#polje oznacava par (vrsta,kolona), a vrednost oznacava koja kolicina je prevezena kroz to polje, u pocetnom resenju
            #polja nam predstavljaju bazne promenljive
            if nebazna:
                nove_cene.append(((i, j), u[i] + v[j] - cena))
    
    return nove_cene

def da_li_poboljsanje(nove_cene):
    for grana, cena in nove_cene:
        if cena > 0: return True
    return False

#ako moze, potrebno je naci novu cenu koja ima najvecu vrednost
def najveca_nova_cena(nove_cene):
    najveca_cena=0;
    
    for grana, cena in nove_cene:
      if cena>najveca_cena:
        najveca_cena=cena
        najveca_cena_vrsta_kolona=grana
    return najveca_cena_vrsta_kolona
    


def moguci_cvorovi(kontura, neposeceni):
    poslednji_cvor = kontura[-1]
    cvorovi_u_redu = [n for n in neposeceni if n[0] == poslednji_cvor[0]]
    cvorovi_u_koloni = [n for n in neposeceni if n[1] == poslednji_cvor[1]]
    if len(kontura) < 2:
        return cvorovi_u_redu + cvorovi_u_koloni
    else:
        prethodni_cvor = kontura[-2]
        pomeren_red = prethodni_cvor[0] == poslednji_cvor[0]
        if pomeren_red: return cvorovi_u_koloni
        return cvorovi_u_redu
    
def napravi_konturu(bv_pozicije, ev_pozicija):
    def inner(kontura):
        if len(kontura) > 3:
            moze_biti_zatvorena = len(moguci_cvorovi(kontura, [ev_pozicija])) == 1
            if moze_biti_zatvorena: return kontura
        
        neposeceni = list(set(bv_pozicije) - set(kontura))
        moguci_sledeci_cvorovi = moguci_cvorovi(kontura, neposeceni)
        for sledeci_cvor in moguci_sledeci_cvorovi:
            nova_kontura = inner(kontura + [sledeci_cvor])
            if nova_kontura: return nova_kontura
    
    return inner([ev_pozicija])


def kontura_pivotiranje(res, kontura):
    parne_celije = kontura[0::2]
    neparne_celije = kontura[1::2]
    get_bv = lambda pos: next(v for p, v in res if p == pos)
    napustena_pozicija = sorted(neparne_celije, key=get_bv)[0]
    napustena_vrednost = get_bv(napustena_pozicija)
    
    novo_res = []
    for p, v in [bv for bv in res if bv[0] != napustena_pozicija] + [(kontura[0], 0)]:
        if p in parne_celije:
            v += napustena_vrednost
        elif p in neparne_celije:
            v -= napustena_vrednost
        novo_res.append((p, v))
        
    return novo_res


global iteracije
global brojIteracijaGlobal
brojIteracijaGlobal = 0

def transportni_metod(ponuda, potraznja, cene, brojMetode, penali=None):
    balansirana_ponuda, balansirana_potraznja, balansirane_cene = izbalansiraj_problem(
        ponuda, potraznja, cene, penali
    )
    global sTot
    sTot=sum(balansirana_ponuda)
    global dTot
    dTot=sum(balansirana_potraznja)
    s=len(balansirana_ponuda)
    d=len(balansirana_potraznja)
    def inner(res,brojIteracijaGlobal):
        brojIteracijaGlobal +=1
        u, v = izracunaj_potencijale(res,balansirane_cene)
        ws = izracunaj_nove_cene(res,  balansirane_cene, u, v)
        if da_li_poboljsanje(ws):
            ev_pozicija = najveca_nova_cena(ws)
            kontura = napravi_konturu([p for p, v in res], ev_pozicija)
            return inner(kontura_pivotiranje(res, kontura),brojIteracijaGlobal)
        return brojIteracijaGlobal,res
    if(brojMetode==1):
        pocetno_res = severozapadni_ugao(balansirana_ponuda,balansirana_potraznja)
        pocetno_transformisano = dopuniNulama(pocetno_res,len(balansirane_cene),len(balansirane_cene[0]))
        iteracije,bazne_promenljive = inner(pocetno_transformisano,brojIteracijaGlobal)
    elif(brojMetode==2):
        pocetno_res = najmanje_cene(balansirana_ponuda,balansirana_potraznja,balansirane_cene)
        pocetno_transformisano = dopuniNulama(pocetno_res,len(balansirane_cene),len(balansirane_cene[0]))        
        iteracije,bazne_promenljive = inner(pocetno_transformisano,brojIteracijaGlobal)
    elif(brojMetode==3):
        balansirana_ponuda_copy=deepcopy(balansirana_ponuda)
        balansirana_potraznja_copy=deepcopy(balansirana_potraznja)
        balansirane_cene_copy=deepcopy(balansirane_cene)
        pocetno_res = transformResult(vogel(len(balansirana_ponuda_copy),len(balansirana_potraznja_copy),balansirana_ponuda_copy,balansirana_potraznja_copy,balansirane_cene_copy))
        pocetno_transformisano = dopuniNulama(pocetno_res,len(balansirane_cene),len(balansirane_cene[0]))
        iteracije,bazne_promenljive = inner(pocetno_transformisano,brojIteracijaGlobal)
    resenje = np.zeros((len(balansirane_cene), len(balansirane_cene[0])))
    for (i, j), v in bazne_promenljive:
        resenje[i][j] = v

    return iteracije,resenje






def ukupna_cena(cene, resenje):
    uk_cena = 0
    for i, row in enumerate(cene):
        for j, cena in enumerate(row):
            uk_cena += cena * resenje[i][j]
    return uk_cena

def ukupna_cena_inicijalno(cene,resenje):
    uk_cena = 0
    for index,bv in enumerate(resenje):
        i,j=bv[0]
        vrednost = bv[1]
        uk_cena+=cene[i][j]*vrednost
    return uk_cena

def prilagodiZaListu(ponuda):
    lista = []
    split=ponuda.split(',')
    for i in split:
        lista.append(int(i))
    return lista
        
def prilagodiZaMatricu(podaci,m,n): ##m je broj redova a n kolona
    lista=[]
    split=podaci.split(',')
    for i in split:
        lista.append(int(i))
    matrica = np.reshape(lista, (m, n))
    return matrica.tolist()

def inicijalnouMatricu(resenje):
    matrica = []
    m,n=nadjiVelicinu(resenje)
    matrica = np.zeros((m+1, n+1))
    for index,bv in enumerate(resenje):
        i,j=bv[0]
        vrednost=bv[1]
        matrica[i][j]=vrednost
    return matrica.tolist()

from tkinter import *
from  tkinter.ttk import *


def generisiTabelu(res,window,ponuda,potraznja):
    headings=[]
    print(ponuda)
    print(potraznja)
    for i in range(len(res)):
        for j in range(len(res[0])):
            res[i][j]=int(res[i][j])
        
    headings.append("") 
    for i in range(len(res[0])):
        headings.append("P"+str(i+1))
    headings.append("Ponuda")
    #ponudu dodajemo kao poslednji element u svakom redu u tabeli
    temp=len(res)
    print(temp)
    for i in range(temp):
        res[i].append(ponuda[i])
        res[i].insert(0,"P"+str(i+1))


    tree = ttk.Treeview(window, column=headings, show='headings',height=len(res)+1,style="mystyle.Treeview")
    for i in range(len(res[0])):
        tree.column("#"+str(i+1),anchor=CENTER,width=75)
        tree.heading("#"+str(i+1), text=headings[i])
    for i in range(len(res)):
            tree.insert('','end',text="1",values=res[i])
    nova_potraznja=[]
    nova_potraznja.append("Potraznja")
    for i in range(len(potraznja)):
        nova_potraznja.append(potraznja[i])
    nova_potraznja.append(str(sum(potraznja))+"/"+str(sum(ponuda)))
    tree.insert('','end',text="1",values=nova_potraznja) #potraznju dodajemo kao posl red u tabeli
     
    return tree
    
            
        

global ponuda
global potraznja
global cene
global penali




        
class StartWindow:
    global ponuda
    global potraznja
    global cene
    global penali    
    def __init__(self, win):
        self.labelaInicijalna=Label(win, text='Pronadji inicijalno resenje')
        self.labelaOptimalna=Label(win, text='Pronadji optimalno resenje')
        self.btnInicijalna = Button(win, text='Pronadji',command =self.inicijalno)
        self.btnOptimalna=  Button(win, text='Pronadji',command =self.optimalno)
        self.btnInicijalna.place(x=200, y=20)
        self.btnOptimalna.place(x=200, y=50)
        self.labelaInicijalna.place(x=50, y=20)
        self.labelaOptimalna.place(x=50, y=50)
    def inicijalno(self):
        global winInicijalno
        winInicijalno = Toplevel(win)
        self.labelaPonuda=Label(winInicijalno, text='Ponuda')
        self.labelaPotraznja=Label(winInicijalno, text='Potraznja')
        self.labelaCene=Label(winInicijalno, text='Cene')
        self.labelaInicijalno=Label(winInicijalno, text='Inicijalno resenje:')
        self.labelaOdabranaPonuda=Label(winInicijalno, text='Odabrana ponuda')
        self.labelaOdabranaPotraznja=Label(winInicijalno, text='Odabrana potraznja')
        self.labelaOdabraneCene=Label(winInicijalno, text='Odabrane cene')
        self.labelaInicijalnoResenje=Label(winInicijalno, text='')
        self.textPotraznja=Entry(winInicijalno)
        self.textCene=Entry(winInicijalno)
        self.textPonuda=Entry(winInicijalno)
        self.textInicijalno=Entry(winInicijalno)
        self.btnSeverozapad = Button(winInicijalno, text='Severozapadni')
        self.btnNajmanjeCene=  Button(winInicijalno, text='Najmanje cene')
        self.btnVogel = Button(winInicijalno, text='Vogel')
        self.btnDodaj = Button(winInicijalno, text='Dodaj podatke')
        self.labelaPonuda.place(x=20, y=10)
        self.labelaOdabranaPonuda.place(x=290, y=10)
        self.labelaOdabranaPotraznja.place(x=290, y=50)
        self.labelaOdabraneCene.place(x=290, y=90)
        self.labelaInicijalno.place(x=20, y=320)
        self.textPotraznja.place(x=120, y=50)
        self.labelaPotraznja.place(x=20, y=50)
        self.textCene.place(x=120, y=90)
        self.labelaInicijalnoResenje.place(x=180, y=320)
        self.btnSeverozapad=Button(winInicijalno, text='Severozapadni ', command=self.racunajSeverozapadni)
        self.btnNajmanjeCene=Button(winInicijalno, text='Najmanje cene',command=self.racunajNajmanjeCene)
        self.btnVogel=Button(winInicijalno, text='Vogel',command=self.racunajVogel)
        self.btnDodaj=Button(winInicijalno, text='Dodaj',command=self.dodajPodatke)
      #  self.btnNajmanjeCene.bind('<Button-1>', self.sub)
        self.btnSeverozapad.place(x=20, y=260)
        self.btnNajmanjeCene.place(x=120, y=260)
        self.btnVogel.place(x=220, y=260)
        self.btnDodaj.place(x=20, y=190)
       # self.btnDodaj.bind('<Button-2>', command=self.dodajPodatke)
        self.labelaCene.place(x=20, y=90)
        self.textPonuda.place(x=120, y=10)
        self.labelaPenali=Label(winInicijalno, text='Penali')
        self.labelaOdabraniPenali=Label(winInicijalno, text='Odabrani penali')
        self.textPenali=Entry(winInicijalno)
        self.labelaOdabraniPenali.place(x=290, y=130)
        self.textPenali.place(x=120, y=130)
        self.labelaPenali.place(x=20, y=130)
        self.labelaUkupna=Label(winInicijalno, text='Ukupna cena:')
        self.labelaUkupnaCena=Label(winInicijalno, text='')
        self.labelaUkupna.place(x=20, y=350)
        self.labelaUkupnaCena.place(x=110, y=350)
        winInicijalno.title('Nadji inicijalno resenje')
        winInicijalno.geometry("650x590+10+10")
   
    def optimalno(self):
        global newwin
        newwin = Toplevel(win)
        global var 
        var = IntVar()
        self.labelaPonuda=Label(newwin, text='Ponuda')
        self.labelaPotraznja=Label(newwin, text='Potraznja')
        self.labelaCene=Label(newwin, text='Cene')
        self.labelaPenali=Label(newwin, text='Penali')
        
        self.labelaOptimalno=Label(newwin, text='Optimalno resenje:')
        self.labelaOdabranaPonuda=Label(newwin, text='Odabrana ponuda')
        self.labelaOdabranaPotraznja=Label(newwin, text='Odabrana potraznja')
        self.labelaOdabraneCene=Label(newwin, text='Odabrane cene')
        self.labelaOdabraniPenali=Label(newwin, text='Odabrani penali')

        self.labelaOptimalnoResenje=Label(newwin, text='')
        self.labelaUkupna=Label(newwin, text='Ukupna cena:')
        self.labelaUkupnaCena1=Label(newwin, text='')
        self.labelaBrojIter=Label(newwin, text='Broj iteracija:')
        self.labelaBrojIteracija=Label(newwin, text='')


        self.textPotraznja=Entry(newwin)
        self.textCene=Entry(newwin)
        self.textPonuda=Entry(newwin)
        self.textPenali=Entry(newwin)

        self.textInicijalno=Entry(newwin)
        self.btnDodaj = Button(newwin, text='Dodaj podatke')
        self.labelaPonuda.place(x=20, y=10)
        self.labelaOdabranaPonuda.place(x=290, y=10)
        self.labelaOdabranaPotraznja.place(x=290, y=50)
        self.labelaOdabraneCene.place(x=290, y=90)
        self.labelaOdabraniPenali.place(x=290, y=130)

        self.labelaOptimalno.place(x=20, y=370)
        self.textPotraznja.place(x=120, y=50)
        self.labelaPotraznja.place(x=20, y=50)
        self.labelaPenali.place(x=20, y=130)
        self.textCene.place(x=120, y=90)
        self.textPenali.place(x=120, y=130)
        
        self.labelaOptimalnoResenje.place(x=180, y=370)
        self.labelaUkupna.place(x=20, y=425)
        self.labelaUkupnaCena1.place(x=110, y=425)
        self.btnDodaj=Button(newwin, text='Dodaj',command=self.dodajPodatke)
        self.btnDodaj.place(x=20, y=185)
        self.labelaCene.place(x=20, y=90)
        self.textPonuda.place(x=120, y=10)
        self.labelaBrojIter.place(x=20, y=460)
        self.labelaBrojIteracija.place(x=140, y=460)

        self.labelaUkupna.place(x=20, y=425)
        
        self.opcijaSeverozapadni = Radiobutton(newwin,text="Severozapadni metod", variable=var, value=1)
        self.opcijaNajmanjeCene = Radiobutton(newwin,text="Metoda najmanjih cena", variable=var, value=2)
        self.opcijaVogel = Radiobutton(newwin,text="Vogelov metod", variable=var, value=3) 
        self.opcijaSeverozapadni.place(x=20, y=250)
        self.opcijaNajmanjeCene.place(x=20, y=280)
        self.opcijaVogel.place(x=20, y=310)
        self.labelaOdabirMetode=Label(newwin, text='Odaberi metodu')
        self.labelaOdabirMetode.place(x=20, y=220)
        self.btnNadjiOptimalno=Button(newwin, text='Izracunaj',command=self.racunajOptimalno)
        self.btnNadjiOptimalno.place(x=20, y=500)
        

        newwin.title('Nadji optimalno resenje')
        newwin.geometry("580x690+10+10")

    def dodajPodatke(self):
        ponuda = str(self.textPonuda.get())
        potraznja = str(self.textPotraznja.get())
        cene = str(self.textCene.get())
        penali = str(self.textPenali.get())
        if len(self.textPonuda.get()) == 0 or len(self.textPotraznja.get()) == 0 or len(self.textCene.get()) == 0 or len(self.textPenali.get()) == 0:
            messagebox.showerror(title="Greska", message="Morate popuniti sva polja")
        self.labelaOdabranaPonuda.configure(text="Odabrana ponuda: " + ponuda)
        self.labelaOdabranaPotraznja.configure(text="Odabrana potraznja: " +potraznja)
        self.labelaOdabraneCene.configure(text="Odabrane cene: " +cene)
        self.labelaOdabraniPenali.configure(text="Odabrani penali: " +penali)


    def racunajSeverozapadni(self):
        if len(self.textPonuda.get()) == 0 or len(self.textPotraznja.get()) == 0 or len(self.textCene.get()) == 0 or len(self.textPenali.get()) == 0:
            messagebox.showerror(title="Greska", message="Morate popuniti sva polja")
        ponuda = prilagodiZaListu(str(self.textPonuda.get()))
        potraznja = prilagodiZaListu(str(self.textPotraznja.get()))
        cene = prilagodiZaMatricu(str(self.textCene.get()),len(ponuda),len(potraznja))
        penali = prilagodiZaListu(str(self.textPenali.get()))        
        ponuda,potraznja,cene = izbalansiraj_problem(ponuda, potraznja, cene,penali)        
        res = severozapadni_ugao(ponuda, potraznja)
        self.labelaInicijalnoResenje.configure(text=str(res))
        self.labelaUkupnaCena.configure(text=str(ukupna_cena_inicijalno(cene,res)))
        res= inicijalnouMatricu(res)
        tree = generisiTabelu(res,winInicijalno,ponuda,potraznja)
        tree.place(x=20,y=400)        
    
    def racunajNajmanjeCene(self):
        if len(self.textPonuda.get()) == 0 or len(self.textPotraznja.get()) == 0 or len(self.textCene.get()) == 0 or len(self.textPenali.get()) == 0:
            messagebox.showerror(title="Greska", message="Morate popuniti sva polja")
        ponuda = prilagodiZaListu(str(self.textPonuda.get()))
        potraznja = prilagodiZaListu(str(self.textPotraznja.get()))
        cene = prilagodiZaMatricu(str(self.textCene.get()),len(ponuda),len(potraznja))
        penali = prilagodiZaListu(str(self.textPenali.get()))        
        ponuda,potraznja,cene = izbalansiraj_problem(ponuda, potraznja, cene,penali) 
        res = najmanje_cene(ponuda, potraznja, cene)
        self.labelaInicijalnoResenje.configure(text=str(res))
        self.labelaUkupnaCena.configure(text=str(ukupna_cena_inicijalno(cene,res)))
        res= inicijalnouMatricu(res)
        tree = generisiTabelu(res,winInicijalno,ponuda,potraznja)
        tree.place(x=20,y=400)
        
    def racunajVogel(self):
        if len(self.textPonuda.get()) == 0 or len(self.textPotraznja.get()) == 0 or len(self.textCene.get()) == 0 or len(self.textPenali.get()) == 0:
            messagebox.showerror(title="Greska", message="Morate popuniti sva polja")
        ponuda = prilagodiZaListu(str(self.textPonuda.get()))
        potraznja = prilagodiZaListu(str(self.textPotraznja.get()))
        cene = prilagodiZaMatricu(str(self.textCene.get()),len(ponuda),len(potraznja))
        penali = prilagodiZaListu(str(self.textPenali.get()))        
        ponuda,potraznja,cene = izbalansiraj_problem(ponuda, potraznja, cene,penali) 
        cene_temp = deepcopy(cene)
        ponuda_temp = deepcopy(ponuda)
        potraznja_temp=deepcopy(potraznja)
        #ponuda,potraznja,cene = izbalansiraj_problem(ponuda, potraznja, cene, [1,1,1,1])
        global sTot
        sTot=sum(ponuda)
        global dTot
        dTot=sum(potraznja)
        s=len(ponuda)
        d=len(potraznja)

        tabela = vogel(s,d,ponuda,potraznja,cene)
        res = transformResult(tabela)
        self.labelaInicijalnoResenje.configure(text=str(res))    
        self.labelaUkupnaCena.configure(text=str(ukupna_cena_inicijalno(cene_temp,res)))
        res= inicijalnouMatricu(res)
        tree = generisiTabelu(res,winInicijalno,ponuda_temp,potraznja_temp)
        tree.place(x=20,y=400)


    def racunajOptimalno(self):
        # Add a Treeview widget
        if len(self.textPonuda.get()) == 0 or len(self.textPotraznja.get()) == 0 or len(self.textCene.get()) == 0 or len(self.textPenali.get()) == 0:
            messagebox.showerror(title="Greska", message="Morate popuniti sva polja")
       # self.labelaOptimalnoResenje.configure(text=str(brojMetode))
        ponuda = prilagodiZaListu(str(self.textPonuda.get()))
        potraznja = prilagodiZaListu(str(self.textPotraznja.get()))
        cene = prilagodiZaMatricu(str(self.textCene.get()),len(ponuda),len(potraznja)) 
        penali = prilagodiZaListu(str(self.textPenali.get()))
        poc_cene = deepcopy(cene)
        if penali=="":
            penali=None
        brojMetode=var.get()
        print(penali)
        br_iter,res = transportni_metod(ponuda,potraznja,cene,brojMetode,penali)
        uk_cena= ukupna_cena(poc_cene, res)
        self.labelaOptimalnoResenje.configure(text=str(res))
        self.labelaUkupnaCena1.configure(text=str(uk_cena))

        self.labelaBrojIteracija.configure(text=str(br_iter))
        
        res=res.tolist()
        #ovde je sekcija za pravljenje tabele
        print(res)
        ponuda,potraznja,cene = izbalansiraj_problem(ponuda, potraznja, cene,penali)
        tree = generisiTabelu(res,newwin,ponuda,potraznja)
        tree.place(x=20,y=540)
        


    




                

win=Tk()
mywin=StartWindow(win)
win.title('Odaberi izracunavanje')
win.geometry("300x100+10+10")
win.option_add( "*font", "Courier 9 bold italic" )

s = ttk.Style()
s.theme_use('xpnative')
s.configure("mystyle.Treeview", font=('Courier', 9,'bold','italic')) # Modify the font of the headings
s.configure("mystyle.Treeview",fieldbackground="black")
win.mainloop()        

