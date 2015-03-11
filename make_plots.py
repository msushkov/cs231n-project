import matplotlib.pyplot as plt
import numpy as np

# data

# lists of 5 accuracies, one for each class
model1 = [0.788732, 0.620795, 0.802773, 0.857600, 0.790385]
model2 = [0.730829, 0.649847, 0.688752, 0.804800, 0.750000]
model3 = [0.503912, 0.669725, 0.580894, 0.788800, 0.759615]

ind = ind = np.arange(5)
width = 0.20

fig, ax = plt.subplots()
rects1 = ax.bar(ind, model1, width, color='r')
rects2 = ax.bar(ind + width, model2, width, color='g')
rects3 = ax.bar(ind + 2 * width, model3, width, color='b')

ax.set_ylabel('CNN classification top-1 accuracy', size=18)
#ax.set_title('Scores by group and gender')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Bicycle', 'Xmas stocking', 'Harp', 'Persian cat', 'Soccer ball'), size=15 )

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Model 1 (AlexNet)', 'Model 2 (finetuned AlexNet)', 'Model 3'), loc=4)

plt.show()


'''
model 2

True label bicycle -> # test points = 639, acc = 0.730829; prec = 0.857143; recall = 0.
637602
  True label christmasstocking -> # test points = 654, acc = 0.649847; prec = 0.570896; r
ecall = 0.573034
  True label harp -> # test points = 649, acc = 0.688752; prec = 0.805405; recall = 0.473
016
  True label persiancat -> # test points = 625, acc = 0.804800; prec = 0.642424; recall =
 0.627219
  True label soccerball -> # test points = 520, acc = 0.750000; prec = 0.610169; recall =
 0.251748
Total # of test points = 3087

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

 True label bicycle -> # test points = 639, acc = 0.788732; prec = 0.808219; recall = 0.
749206
  True label christmasstocking -> # test points = 654, acc = 0.620795; prec = 0.192308; r
ecall = 0.046296
  True label harp -> # test points = 649, acc = 0.802773; prec = 0.778210; recall = 0.738
007
  True label persiancat -> # test points = 625, acc = 0.857600; prec = 0.613636; recall =
 0.837209
  True label soccerball -> # test points = 520, acc = 0.790385; prec = 0.142857; recall =
 0.009615
Total # of test points = 3087

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

 True label bicycle -> # test points = 639, acc = 0.503912; prec = -1.000000; recall = 0
.000000
  True label christmasstocking -> # test points = 654, acc = 0.669725; prec = 1.000000; r
ecall = 0.004608
  True label harp -> # test points = 649, acc = 0.580894; prec = -1.000000; recall = 0.00
0000
  True label persiancat -> # test points = 625, acc = 0.788800; prec = -1.000000; recall
= 0.000000
  True label soccerball -> # test points = 520, acc = 0.759615; prec = -1.000000; recall
= 0.000000
Total # of test points = 3087

{'harp/harp_0586.jpg': ('FN', 'harp', ['trash']), 'harp/harp_0460.jpg': ('FN', 'harp', ['
trash']), 'harp/harp_0000.jpg': ('FN', 'harp', ['trash']), 'bicycle/bicycle_0392.jpg': ('
FN', 'bicycle', ['trash']), 'persiancat/cat_0343.jpg': ('FN', 'persiancat', ['trash']), '
christmasstocking/christmasstocking_0093.jpg': ('FN', 'christmasstocking', ['trash']), 'h
arp/harp_0654.jpg': ('FN', 'harp', ['trash']), 'harp/harp_0152.jpg': ('FN', 'harp', ['tra
sh']), 'soccerball/soccerball_0122.jpg': ('FN', 'soccerball', ['trash']), 'harp/harp_0354
.jpg': ('FN', 'harp', ['trash'])}


'''