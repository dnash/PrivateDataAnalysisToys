import random

towrite='malignant,diameter,height\n'

for i in range(1000):
    thisline=random.choice([0,1])
    if thisline==0:
        diameter=random.gauss(2,0.5)
        height=random.gauss(1,0.25)
        #Overlapping:
        #diameter=random.gauss(2,1.5)
        #height=random.gauss(1,1.25)
        if random.choice([0,1])==0:
            diameter=random.gauss(2,0.5)
            height=random.gauss(1,0.25)
        else:
            diameter=random.gauss(5,0.5)
            height=random.gauss(1,0.25)        
    if thisline==1:
        diameter=random.gauss(3.5,0.5)
        height=random.gauss(1.5,0.35)
        #Overlapping:
        #diameter=random.gauss(3,2.5)
        #height=random.gauss(1.5,1.25)
    towrite+=str(thisline)+","+str(round(diameter,3))+","+str(round(height,3))+'\n'

thisfile=open('DummyData4.csv','w')
thisfile.write(towrite)
thisfile.close()
