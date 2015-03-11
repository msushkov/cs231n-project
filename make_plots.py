import matplotlib.pyplot as plt

plt.figure(1)
plt.xlabel('Path Length When Successful', size=20)
plt.ylabel('Frequency', size=20)

# data

# lists of 5 accuracies, one for each class
model1 = [0.788732, 0.620795, 0.802773, 0.857600, 0.790385]
model2 = [0.730829, 0.649847, 0.688752, 0.804800, 0.750000]
model3 = []

'''
model 2

{'bicycle/bicycle_0515.jpg': ('FP', 'bicycle', ['bicycle']), 'christmasstocking/christmas
stocking_0358.jpg': ('FN', 'christmasstocking', ['trash']), 'bicycle/bicycle_0296.jpg': (
'FN', 'bicycle', ['trash']), 'harp/harp_0000.jpg': ('FN', 'harp', ['trash']), 'persiancat
/cat_0308.jpg': ('FP', 'persiancat', ['persiancat']), 'harp/harp_0195.jpg': ('FN', 'harp'
, ['trash']), 'christmasstocking/christmasstocking_0462.jpg': ('FN', 'christmasstocking',
 ['trash']), 'christmasstocking/christmasstocking_0024.jpg': ('FP', 'christmasstocking',
['christmasstocking']), 'christmasstocking/christmasstocking_0031.jpg': ('FP', 'christmas
stocking', ['christmasstocking']), 'christmasstocking/christmasstocking_0054.jpg': ('FP',
 'christmasstocking', ['christmasstocking']), 'christmasstocking/christmasstocking_0175.j
pg': ('FN', 'christmasstocking', ['trash']), 'christmasstocking/christmasstocking_0624.jp
g': ('FP', 'christmasstocking', ['christmasstocking']), 'harp/harp_0265.jpg': ('FP', 'har
p', ['harp']), 'bicycle/bicycle_0385.jpg': ('FN', 'bicycle', ['trash']), 'christmasstocki
ng/christmasstocking_0628.jpg': ('FP', 'christmasstocking', ['christmasstocking']), 'bicy
cle/bicycle_0348.jpg': ('FP', 'bicycle', ['bicycle']), 'christmasstocking/christmasstocki
ng_0583.jpg': ('FN', 'christmasstocking', ['trash']), 'christmasstocking/christmasstockin
g_0008.jpg': ('FN', 'christmasstocking', ['trash']), 'christmasstocking/christmasstocking
_0301.jpg': ('FP', 'christmasstocking', ['christmasstocking']), 'soccerball/soccerball_03
19.jpg': ('FN', 'soccerball', ['trash'])}


model 1

{'bicycle/bicycle_0515.jpg': ('FP', 'bicycle', ['bicycle', 'trash']), 'bicycle/bicycle_04
01.jpg': ('FP', 'bicycle', ['bicycle', 'trash']), 'christmasstocking/christmasstocking_04
93.jpg': ('FN', 'christmasstocking', ['trash']), 'persiancat/cat_0308.jpg': ('FP', 'persi
ancat', ['trash', 'persiancat']), 'harp/harp_0515.jpg': ('FP', 'harp', ['trash', 'harp'])
, 'harp/harp_0195.jpg': ('FN', 'harp', ['trash', 'christmasstocking']), 'harp/harp_0653.j
pg': ('FP', 'harp', ['trash', 'harp']), 'christmasstocking/christmasstocking_0093.jpg': (
'FN', 'christmasstocking', ['trash']), 'christmasstocking/christmasstocking_0175.jpg': ('
FN', 'christmasstocking', ['trash']), 'christmasstocking/christmasstocking_0358.jpg': ('F
N', 'christmasstocking', ['trash']), 'christmasstocking/christmasstocking_0121.jpg': ('FP
', 'christmasstocking', ['trash', 'christmasstocking']), 'soccerball/soccerball_0139.jpg'
: ('FN', 'soccerball', ['trash']), 'christmasstocking/christmasstocking_0628.jpg': ('FP',
 'christmasstocking', ['trash', 'christmasstocking']), 'bicycle/bicycle_0348.jpg': ('FP',
 'bicycle', ['bicycle', 'trash']), 'christmasstocking/christmasstocking_0540.jpg': ('FP',
 'christmasstocking', ['trash', 'christmasstocking']), 'bicycle/bicycle_0097.jpg': ('FP',
 'bicycle', ['bicycle', 'trash', 'harp']), 'bicycle/bicycle_0443.jpg': ('FN', 'bicycle',
['trash', 'harp']), 'soccerball/soccerball_0122.jpg': ('FN', 'soccerball', ['trash']), 's
occerball/soccerball_0319.jpg': ('FN', 'soccerball', ['trash']), 'christmasstocking/chris
tmasstocking_0592.jpg': ('FN', 'christmasstocking', ['trash'])}

model 3



'''

n, bins, patches = plt.hist([model1, model2, model3], 3, normed=0, label=["Model 1", "Model 2", "Model 3"])    

plt.legend(loc=1)
plt.show()