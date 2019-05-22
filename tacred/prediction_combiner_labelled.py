from sklearn.cluster import DBSCAN,AgglomerativeClustering,MeanShift
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression as LR
from sklearn.isotonic import IsotonicRegression as IR
from sklearn import metrics
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.backends.backend_pdf

import numpy
import os
import sys

def binom_interval(success, total, confint=0.90):
    quantile = (1 - confint) / 2.
    lower = beta.ppf(quantile, success, total - success + 1)
    upper = beta.ppf(1 - quantile, success + 1, total - success)
    return (lower, upper)

def get_precision_and_recall(probabilities,classes):
    classes = numpy.array(classes)
    probabilities = numpy.array(probabilities)
    positive = 0
    negative = 0
    for c in classes:
        if c==1:
            positive+=1
        else:
            negative+=1
    tpa=0
    tpb = 0
    tna = negative
    tnb = negative
    fpa = 0
    fpb = 0
    fna = positive
    fnb = positive
    
    
    plist = []
    rlist = []
    llist = []
    ulist = []
    er = []
    ep = []
    
    sorted_prob = probabilities.argsort()[::-1][:]
    #print(sorted_prob)
    current = classes[sorted_prob[0]]
    precision = 0
    recall = 0
    count = 0
    for s in sorted_prob:
        count+=1
        if classes[s] != current:
            current = classes[s]
            x = abs(tpb-tpa)/float(2)
            if x != 0:
                old_recall = tpa/float(tpa+fna)
                new_recall = tpb/float(tpb+fnb)
                half_recall = new_recall-old_recall/2
                
                #old_precision = tpa/float(tpa+fpa)
                new_precision = tpb/float(tpb+fpb)
                
                half_precision = float(tpa+x)/(tpa+x+fpa+x*(fpb-fpa)/(tpb-tpa))
                plist.append(half_precision)
                plist.append(new_precision)
                rlist.append(half_recall)
                rlist.append(new_recall)
                if count %10 == 0:
                    lower,upper = binom_interval(tpb,tpb+fpb)
                    lower = precision-lower
                    upper = upper-precision
                    llist.append(lower)
                    ulist.append(upper)
                    er.append(recall)
                    ep.append(precision)
    
    
        if classes[s] == 1:
            tpb+=1
            fnb-=1
        else:
            fpb+=1
            tnb-=1
        precision = float(tpb)/(tpb+fpb)
        recall = float(tpb)/(tpb+fnb)

    plist.append(precision)
    rlist.append(recall)
    return plist,rlist,llist,ulist,er,ep


def build_dataset(filename):
    file = open(filename,'rU')
    lines = file.readlines()
    file.close()
    
    data_dict ={}
    group_set = set()
    for i in range(1,len(lines)):
        split_line = lines[i].split('\t')
        data_dict[i-1] = split_line[:-1]
        group_set.add(split_line[-1].strip())
    return data_dict, group_set

def noisy_or(probabilities):
    print('initial')
    print(probabilities)
    negation_prob = 1 - numpy.array(probabilities)
    print(negation_prob)
    noisy_or = 1-numpy.prod(negation_prob)
    print('final')
    print(noisy_or)
    return noisy_or

def calibrate_probs(probabilities,classes):
    ir = IR(out_of_bounds='clip')
    ir.fit(probabilities,classes) #fit ir to abstract level precision and classes
    p_calibrated=ir.transform(probabilities)
    
    return p_calibrated

def calibrate_dictionary(prob_dict):
    
    labels = []
    probabilities = []
    for i in range(len(prob_dict)):
        labels.append(float(prob_dict[i][3]))
        probabilities.append(float(prob_dict[i][4]))
    
    p_calibrated = calibrate_probs(probabilities,labels)
    for p in range(len(p_calibrated)):
        prob_dict[p][4]=p_calibrated[p]
    return prob_dict

def cluster_gradients(gradients):
    if len(gradients) == 1:
        return [0]
    clustering = DBSCAN(eps=1.0,min_samples=1).fit(gradients)
    return clustering.labels_

def map_clusters(data_dict,grouping,cluster_labels):
    cluster_dict = {}
    for i in range(len(cluster_labels)):
        if cluster_labels[i] not in cluster_dict:
            cluster_dict[cluster_labels[i]] = []
        cluster_dict[cluster_labels[i]].append(float(data_dict[grouping[i]][4]))
    
    #print(cluster_dict)
    cluster_max_probs = []
    for cluster in cluster_dict:
        cluster_max_probs.append(max(cluster_dict[cluster]))
    return cluster_max_probs

def cluster_noisy_or(data_dict,grouping):
    grad_list = []
    label = 0
    label = ()
    for g in grouping:
        grad_list.append([float(x) for x in data_dict[g][5].split('|')])
        label = (data_dict[g][1],data_dict[g][2])
    cluster_labels = cluster_gradients(grad_list)
    cluster_max_probs = map_clusters(data_dict,grouping,cluster_labels)
    noisy_or_prob = noisy_or(cluster_max_probs)
    return noisy_or_prob,label

def single_noisy_or(data_dict,grouping):
    prob_list = []
    label = 0
    entities = ()
    for g in grouping:
        prob_list.append(float(data_dict[g][4]))
        entities = (data_dict[g][1],data_dict[g][2])
    noisy_or_prob = noisy_or(numpy.array(prob_list))
    return noisy_or_prob,entities

def noisy_or_builders(data_dict,group_dict):
    single_probs = []
    single_class =[]
    cluster_probs = []
    cluster_class = []
    for group in group_dict:
        format_group = [int(a) for a in group.split('|')]
        prob, label = single_noisy_or(data_dict,format_group)
        cluster_prob,cluster_label = cluster_noisy_or(data_dict,format_group)
        single_probs.append(prob)
        single_class.append(label)
        cluster_probs.append(cluster_prob)
        cluster_class.append(cluster_label)
    
    return single_probs,single_class,cluster_probs,cluster_class

def generate_precision_recall_curves_with_error(probs,labels):
    plist,rlist,llist,ulist,er,ep = get_precision_and_recall(probs,labels)
    
    plt.plot(rlist,plist,linewidth=2)
    
    l = numpy.array(llist)
    u = numpy.array(ulist)
    asym = [l,u]
    rx = numpy.array(er)
    py = numpy.array(ep)
    plt.errorbar(rx, py, yerr=asym,color='black',fmt='none')
    
    return

def generate_precision_recall_curves(probs,labels):
    plist,rlist,llist,ulist,er,ep = get_precision_and_recall(probs,labels)
    plt.plot(rlist,plist,linewidth=2)
    
    return


def build_noisy(filename,noisy_or_dict,noisy_label):
    file = open(filename,'rU')
    lines = file.readlines()
    file.close()

    for l in range(1,len(lines)):
        split_line = lines[l].split('\t')
        entity_1 = split_line[0]
        entity_2 = split_line[1]
        if (entity_1,entity_2) not in noisy_or_dict:
            noisy_or_dict[(entity_1,entity_2)] = []
            noisy_label[(entity_1,entity_2)] = 0
        noisy_or_dict[(entity_1,entity_2)].append(float(split_line[2].strip()))
        noisy_label[(entity_1,entity_2)] = max(noisy_label[(split_line[0],split_line[1])],int(float(split_line[3].strip())))
    return noisy_or_dict,noisy_label

def main():
    noisy_or_dict = {}
    noisy_label = {}
    for i in range(1,len(sys.argv)-1):
        noisy_or_dict,noisy_label = build_noisy(sys.argv[i],noisy_or_dict,noisy_label)
    
    total_dict = {}
    for val in noisy_or_dict:
        total_dict[val] = noisy_or(noisy_or_dict[val])

    outfile = open(sys.argv[-1],'w')
    outfile.write('ENTITY1\tENTITY2\tPROBABILITY\tLABEL\n')
    for val in sorted(total_dict, key=total_dict.get,reverse=True):
        outfile.write(val[0]+'\t'+val[1]+'\t'+str(noisy_or(noisy_or_dict[val])) +'\t'+str(noisy_label[val])+'\n')
    outfile.close()


if __name__ == '__main__':
    main()
