import numpy
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.backends.backend_pdf
from scipy.stats import beta
from sklearn import metrics

#clopper_pearson
def binom_interval(success, total, confint=0.90):
    print(total)
    print(success)
    q = (1-confint)/2.0
    lower = beta.ppf(q, success, total - success + 1)
    print(lower)
    upper = beta.ppf(1-q, success + 1, total - success)
    print(upper)
    if success == total:
        upper = 1
    if success == 0:
        lower = 0
    return (lower, upper)

def get_lines(filename):
    file = open(filename,'rU')
    lines = file.readlines()

    file.close()
    positive = 0
    negative = 0
    for l in range(1,len(lines)):
        line = lines[l].split()
            #if float(line[0]) == 0:
            #break
        if line[3].strip() == '1':
            positive +=1
        #print(positive)
        else:
            negative +=1

    #print(positive)
    #print(negative)
    accuracy = float(positive)/(positive+negative)

    fp = 0
    tp = 0
    fn = positive
    print('fn')
    print(fn)
    tn = negative
    print(tn)
    plist = [0]
    rlist = [0]
    llist = []
    ulist = []
    er = []
    ep = []
    
    current = '0'
    for l in range(1,len(lines)):
        interval_size = len(lines)/10
        line = lines[l].split()
        if float(line[2]) == 0:
            break
        if line[3] == '1':
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1
        if line[3] != current:
            current = line[3]
            if current == '0':
                precision = float(tp)/(tp+fp-1)
                #print(precision)
                plist.append(precision)
                recall = float(tp)/(tp+fn)
                #print(recall)
                rlist.append(recall)
                if len(rlist)%5== 0:
                    lower,upper = binom_interval(tp,tp+fp-1)
                    lower = precision-lower
                    upper = upper-precision
                    llist.append(lower)
                    ulist.append(upper)
                    er.append(recall)
                    ep.append(precision)

    if len(plist)>1:
        plist[0]=plist[1]

    return rlist, plist, llist,ulist,er,ep,accuracy
#print(str(recall) + '\t' + str(precision))

key_order = sys.argv[1:]
total_key_order = []
for x in key_order:
    total_key_order.append(os.path.splitext(os.path.basename(x))[0])

plt.figure()
rlist,plist,llist,ulist,er,ep,accuracy = get_lines(sys.argv[1])
l = numpy.array(llist)
u = numpy.array(ulist)
asym = [l,u]
x = numpy.array(rlist)
print(x)
y = numpy.array(plist)
#print(y)
print('auc')
#print(metrics.auc(x,y))
plt.plot(x, y,linewidth=1,color='blue')
print(x)
rx = numpy.array(er)
py = numpy.array(ep)
plt.errorbar(rx, py, yerr=asym,ls='none',color='blue',capsize=2)

colors = ['blue','red','teal','orange']
if len(sys.argv)>=3:
    print('hi')
    for i in range(2,len(sys.argv)):
        rlist,plist,llist,ulist,er,ep,accuracy = get_lines(sys.argv[i])
        x = numpy.array(rlist)
        #print(x)
        y = numpy.array(plist)
        #print(y)
        print('auc')
        #print(metrics.auc(x,y))
        plt.plot(x, y,linewidth=2,color=colors[i-1])
        plt.axis([0,1.01,0,1.01])
#red_patch = mpatches.Patch(color='red', label='The red data')
#blue_patch = mpatches.Patch(color='blue',label='The blue data')


print(accuracy)
plt.plot([0,1],[accuracy,accuracy],'r--',lw=2,label='Baseline',color='green')
total_key_order = ['Markov Network', 'Noisy-OR']

plt.figure(1)
#plt.title('cross validated precision-recall curve for HIV-1',fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Recall',fontsize=18)
plt.ylabel('Precision',fontsize=18)
plt.legend(total_key_order, loc='best',fontsize=18)
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])

#plt.savefig('figure1.png')
#precisionlist = list(reversed(plist))
#recalllist = list(reversed(rlist))

#pdf = matplotlib.backends.backend_pdf.PdfPages("binds.pdf")
#for fig in xrange(1, plt.gcf().number+1): ## will open an empty extra figure :(
#    pdf.savefig( fig )
#pdf.close()

plt.savefig('tacred/tacred_gradient_calibrated.png')

