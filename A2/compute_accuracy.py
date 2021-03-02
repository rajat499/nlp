import re
import sys
from sklearn import metrics
classes = ['A', 'C', 'L', 'LA', 'N', 'O', 'P', 'T']

#gold labels
with open(sys.argv[1], "r") as f:
	st = f.read().strip()
	st = re.sub(r'\n\s*\n', '\n\n', st)

lines = st.split("\n\n")
gold = [line.split('\n') for line in lines]
goldlabels = []
for sent in gold:
	sentlabels = [item.split(" ")[1] for item in sent]
	goldlabels += sentlabels

#predictions
with open(sys.argv[2], "r") as f:
	st = f.read().strip()
lines = st.split("\n\n")
predlabels = []
for line in lines:
	predlabels += line.split("\n")


#report
report = metrics.classification_report(goldlabels, predlabels, output_dict=True)
print("Classwise F-1 scores: \n")
for key in classes:
	print(key + ": " + str(report[key]['f1-score']))
