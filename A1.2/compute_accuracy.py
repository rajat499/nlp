import csv
import sys

label_to_num = {'Meeting and appointment':'1', 'For circulation':'2', 'Selection committe issues':'3', 'Policy clarification/setting':'4', 'Recruitment related': '5', 'Assessment related':'6', 'Other':'7'}
csv.field_size_limit(sys.maxsize)

with open(sys.argv[2], "r") as f:
    pred = f.read().splitlines()

gold = []
with open(sys.argv[1], "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for j,row in enumerate(csv_reader):
        if j == 0:
            continue
        else:
            gold.append(label_to_num[row[2]])
print(gold)
print(pred)
#compute Micro-accuracy:
micro = (sum(1 for j in range(len(gold)) if gold[j] == pred[j]))/len((gold))
print("Micro-accuracy = " + str(micro))

#compute Macro-accuracy:
avg = 0.0
for i in range(1,8):
	class_tot = sum(1 for j in range(len(gold)) if gold[j] == str(i))
	class_corr = sum(1 for j in range(len(gold)) if gold[j] == str(i) and gold[j] == pred[j])
	avg += class_corr/class_tot
macro = avg/7
print("Macro-accuracy = " + str(macro))
