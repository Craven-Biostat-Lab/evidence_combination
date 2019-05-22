import numpy
import os
import sys
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.backends.backend_pdf
import operator as op
import random
from functools import reduce
from sklearn.cluster import DBSCAN,AgglomerativeClustering,MeanShift
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression as LR
from sklearn.isotonic import IsotonicRegression as IR
from sklearn import metrics
from scipy.stats import beta

random.seed(2500)

def kcr(k, r):
    r = min(r, k-r)
    numer = reduce(op.mul, range(k, k-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom


def calc_prob_0(prev_prob,current_prob,cos_sim_i_j,prev_alpha1,prev_alpha0,n,k,r):
    print('alpha_0_calc')
    #print(prev_alpha0,prev_alpha1)
    term0 = zero_zero(prev_prob,current_prob,cos_sim_i_j,n,k,r)*prev_alpha0
    term1 = one_zero(prev_prob,current_prob,cos_sim_i_j,n,k,r)*prev_alpha1
    print(term0,term1)
    final_prob = term0+term1
    return final_prob

def calc_prob_1(prev_prob,current_prob,cos_sim_i_j,prev_alpha1,prev_alpha0,n,k,r):
    print('alpha_1_calc')
    #print(prev_alpha0,prev_alpha1)
    term0 = zero_one(prev_prob,current_prob,cos_sim_i_j,n,k,r)*prev_alpha0
    term1 = one_one(prev_prob,current_prob,cos_sim_i_j,n,k,r)*prev_alpha1
    print(term0,term1)
    final_prob = term0+term1
    return final_prob

def zero_zero(prob_i,prob_j,cos_sim_i_j,n,k,r):
    print('0-0')
    print(prob_i,prob_j,cos_sim_i_j)
    val= (max((1-prob_i),(1-prob_j))**(float(1)/(kcr(k,r))))*cos_sim_i_j + (1-cos_sim_i_j)*(((1-prob_i)*(1-prob_j))**(float(1)/(n-1)))
    #val = (1-cos_sim_i_j)*((1-prob_i)*(1-prob_j))**(float(1)/(n-1))
    print(val)
    return val
def zero_one(prob_i,prob_j,cos_sim_i_j,n,k,r):
    print('0-1')
    print(prob_i,prob_j,cos_sim_i_j)
    val = (1-cos_sim_i_j)*((1-prob_i)*prob_j)**(float(1)/(n-1))
    print(val)
    return val

def one_zero(prob_i,prob_j,cos_sim_i_j,n,k,r):
    print('1-0')
    print(prob_i,prob_j,cos_sim_i_j)
    val =  (1-cos_sim_i_j)*(prob_i*(1-prob_j))**(float(1)/(n-1))
    print(val)
    return val

def one_one(prob_i,prob_j,cos_sim_i_j,n,k,r):
    print('1-1')
    print(prob_i,prob_j,cos_sim_i_j)
    val =  (max(prob_i,prob_j)**(float(1)/(kcr(k,r))))*cos_sim_i_j + (1-cos_sim_i_j)*(((prob_i)*(prob_j))**(float(1)/(n-1)))
    #val = (1-cos_sim_i_j)*(prob_i*prob_j)**(float(1)/(n-1))
    print(val)
    return(val)


def calibrate_probabilities(prob_dict,instance_label_dict):
    labels = []
    probabilities = []
    print(len(prob_dict))
    print(len(instance_label_dict))
    for i in prob_dict:
        labels.append(instance_label_dict[i])
        probabilities.append(prob_dict[i])

    ir = IR(out_of_bounds='clip')
    ir.fit(probabilities,labels) #fit ir to abstract level precision and classes
    p_calibrated=ir.transform(probabilities)

    fig,ax = plt.subplots()
    fraction_of_positives, mean_predicted_value = calibration_curve(labels, p_calibrated, n_bins=10)
    ax.plot(mean_predicted_value, fraction_of_positives)
    fraction_of_positives, mean_predicted_value = calibration_curve(labels, probabilities, n_bins=10)
    ax.plot(mean_predicted_value, fraction_of_positives)

    plt.savefig('calibration_curve_on_data.png')
    return ir
               
def get_data(filename):
    count = 0
    file = open(filename,'rU')
    lines = file.readlines()
    file.close()
    group_dict = {}
    prob_dict = {}
    label_dict = {}
    instance_label_dict = {}
    cos_dict = {}
    pmids = set()
    for i in range(1,len(lines)):
        split_line = lines[i].strip().split('\t')
        pmids.add(split_line[0])
    
    pmids=list(pmids)
    random.shuffle(pmids)
    divider = len(pmids)/3
    calibration_set = set(pmids[:divider])
    test_set = set(pmids[divider:])
    
    for i in range(1,len(lines)):
        count+=1
        split_line = lines[i].strip().split('\t')
        if split_line[0] in calibration_set:
            instance = str(i-1)
            entity_1 = split_line[1]
            entity_2 = split_line[2]
            #print(entity_1,entity_2)
            groupings = split_line[6].split('|')
            cosines = split_line[5].split('|')

            group_dict[(entity_1,entity_2)] = groupings
            prob_dict[instance] = float(split_line[4])
            instance_label_dict[instance] = int(float(split_line[3]))
            if (entity_1,entity_2) not in label_dict:
                label_dict[(entity_1,entity_2)] = 0
            label_dict[(entity_1,entity_2)] = max(label_dict[(entity_1,entity_2)],int(float(split_line[3])))
            cos_dict[instance] = {}
            for g in range(len(groupings)):
                #check similarity
                cos_dict[instance][groupings[g]] =  cosines[g]
    print(len(calibration_set))
    print(len(prob_dict))
    ir = calibrate_probabilities(prob_dict,instance_label_dict)
    
    group_dict = {}
    prob_dict = {}
    label_dict = {}
    instance_label_dict = {}
    cos_dict = {}
    
    for i in range(1,len(lines)):
        count+=1
        split_line = lines[i].strip().split('\t')
        if split_line[0] in test_set:
            instance = str(i-1)
            entity_1 = split_line[1]
            entity_2 = split_line[2]
            #print(entity_1,entity_2)
            groupings = split_line[6].split('|')
            cosines = split_line[5].split('|')
            
            group_dict[(entity_1,entity_2)] = groupings
            prob_dict[instance] = float(split_line[4])
            instance_label_dict[instance] = int(float(split_line[3]))
            if (entity_1,entity_2) not in label_dict:
                label_dict[(entity_1,entity_2)] = 0
            label_dict[(entity_1,entity_2)] = max(label_dict[(entity_1,entity_2)],int(float(split_line[3])))
            cos_dict[instance] = {}
            for g in range(len(groupings)):
                #check similarity
                cos_dict[instance][groupings[g]] =  cosines[g]
    print(len(test_set))
    print(len(prob_dict))
    labels = []
    probabilities = []
    index = 0
    index_vals = {}
    for i in prob_dict:
        labels.append(instance_label_dict[i])
        probabilities.append(prob_dict[i])
        index_vals[i] = index
        index+=1


    p_calibrated=ir.transform(probabilities)
    #p_calibrated=probabilities

    calibrated_dict = {}
    for i in index_vals:
        calibrated_dict[i] = p_calibrated[index_vals[i]]

    cal_keys = set(calibrated_dict.keys())
    for g in group_dict:
        group_dict[g] = list(set(group_dict[g]).intersection(cal_keys))
        print(group_dict[g])

    fig,ax = plt.subplots()
    fraction_of_positives, mean_predicted_value = calibration_curve(labels, p_calibrated, n_bins=10)
    ax.plot(mean_predicted_value, fraction_of_positives)
    fraction_of_positives, mean_predicted_value = calibration_curve(labels, probabilities, n_bins=10)
    ax.plot(mean_predicted_value, fraction_of_positives)

    plt.savefig('calibration_curve.png')


    return group_dict,calibrated_dict,label_dict,cos_dict

#n = noisy_or value
#k = cluster size
#r = choose value
def linear_chain(group,group_dict,prob_dict,label_dict,cos_dict):
    
    print(group_dict[group])
    k = len(group_dict[group])
    prev_alpha_0_0 = 1 #prev_alpha_FIRSTINDEXSTATE_CURRENTINDEXSTATE
    prev_alpha_0_1 = 0
    prev_alpha_1_1 = 1
    prev_alpha_1_0 = 0
    first_index = group_dict[group][0]
    last_index = group_dict[group][-1]
    first_prob = prob_dict[first_index]
    last_prob = prob_dict[last_index]
    last_cos = min(1,float(cos_dict[last_index][first_index]))
    

    for i in range(1,len(group_dict[group])):
        current_index = group_dict[group][i]
        prev_index = group_dict[group][i-1]
        prob_i = prob_dict[prev_index]
        prob_j = prob_dict[current_index]
        cos_sim_i_j = min(1,float(cos_dict[current_index][prev_index]))
        new_alpha_0_0 = calc_prob_0(prob_i,prob_j,cos_sim_i_j,prev_alpha_0_1,prev_alpha_0_0,3,k,1)
        new_alpha_1_0 = calc_prob_0(prob_i,prob_j,cos_sim_i_j,prev_alpha_1_1,prev_alpha_1_0,3,k,1)
        new_alpha_0_1 = calc_prob_1(prob_i,prob_j,cos_sim_i_j,prev_alpha_0_1,prev_alpha_0_0,3,k,1)
        new_alpha_1_1 = calc_prob_1(prob_i,prob_j,cos_sim_i_j,prev_alpha_1_1,prev_alpha_1_0,3,k,1)
        prev_alpha_0_0 = new_alpha_0_0
        prev_alpha_0_1 = new_alpha_0_1
        prev_alpha_1_0 = new_alpha_1_0
        prev_alpha_1_1 = new_alpha_1_1
    new_alpha_0_0 = calc_prob_0(last_prob,first_prob,last_cos,prev_alpha_0_1,prev_alpha_0_0,3,k,1)
    new_alpha_1_0 = calc_prob_0(last_prob,first_prob,last_cos,prev_alpha_1_1,prev_alpha_1_0,3,k,1)
    new_alpha_0_1 = calc_prob_1(last_prob,first_prob,last_cos,prev_alpha_0_1,prev_alpha_0_0,3,k,1)
    new_alpha_1_1 = calc_prob_1(last_prob,first_prob,last_cos,prev_alpha_1_1,prev_alpha_1_0,3,k,1)

    partition = new_alpha_0_0 + new_alpha_1_1


    zero_prob = 1
    for i in range(1,len(group_dict[group])):
        current_index = group_dict[group][i]
        prev_index = group_dict[group][i-1]
        prob_i = prob_dict[prev_index]
        prob_j = prob_dict[current_index]
        cos_sim_i_j = min(1,float(cos_dict[current_index][prev_index]))
        zero_prob = zero_prob*zero_zero(prob_i,prob_j,cos_sim_i_j,3,k,1)
    zero_prob = zero_prob*zero_zero(last_prob,first_prob,last_cos,3,k,1)
    final_prob = 1-(zero_prob/partition)
    print('LINEAR_VALUE')
    print('final partition',partition)
    print('zero prob',zero_prob)
    print('final prob',final_prob)
    return final_prob


def pair_wise(group,group_dict,prob_dict,label_dict,cos_dict):
    pairs = list(itertools.combinations(group_dict[group], 2))
    n = len(group_dict[group])
    k = n
    binaries = list(itertools.product([0,1],repeat = len(group_dict[group])))
    partition = 0
    for binary_order in binaries:
        binary_order = list(binary_order)
        binary_pairs = list(itertools.combinations(binary_order, 2))
        single = 1
        for p in range(len(pairs)):
            bp1 = binary_pairs[p][0]
            bp2 = binary_pairs[p][1]
            i1 = pairs[p][0]
            i2 = pairs[p][1]
            prob_i=prob_dict[i1]
            prob_j=prob_dict[i2]
            cos_sim_i_j = min(float(cos_dict[i1][i2]),1)
            if bp1 == 0 and bp2 == 0:
                single*=zero_zero(prob_i,prob_j,cos_sim_i_j,n,k,1)
            elif bp1 == 0 and bp2 == 1:
                single*= zero_one(prob_i,prob_j,cos_sim_i_j,n,k,1)
            elif bp1 == 1 and bp2 == 0:
                single*= one_zero(prob_i,prob_j,cos_sim_i_j,n,k,1)
            else:
                single*=one_one(prob_i,prob_j,cos_sim_i_j,n,k,1)
        partition += single

    zero_prob = 1
    for p in range(len(pairs)):
        i1 = pairs[p][0]
        i2 = pairs[p][1]
        prob_i=prob_dict[i1]
        prob_j=prob_dict[i2]
        cos_sim_i_j = min(1,float(cos_dict[i1][i2]))
        zero_prob*=zero_zero(prob_i,prob_j,cos_sim_i_j,n,k,1)
        
    final_prob = 1-(zero_prob/partition)
    print('pairwise_VALUE')
    print('final partition',partition)
    print('zero prob',zero_prob)
    print('final prob',final_prob)
    return final_prob

def noisy_or(group,group_dict,prob_dict,label_dict,cos_dict):
    probabilities = []
    for i in range(0,len(group_dict[group])):
        current_index = group_dict[group][i]
        prob_i = prob_dict[current_index]
        probabilities.append(prob_i)
    negation_prob = 1- numpy.array(probabilities)
    noisy_or = 1-numpy.prod(negation_prob)
    return noisy_or


def main():
    group_dict,prob_dict,label_dict,cos_dict = get_data(sys.argv[1])
    print('group')
    print(group_dict)
    chain_outfile = open(sys.argv[2],'w')
    chain_outfile.write('E1\tE2\tProbability\tLabel\n')
    outfile = open(sys.argv[3],'w')
    outfile.write('E1\tE2\tProbability\tLabel\n')
    noisy_out = open(sys.argv[4],'w')
    noisy_out.write('E1\tE2\tProbability\tLabel\n')
    
    hybrids = []
    noisy_ors = []
    sizes = []
    labels = []
    for group in group_dict:
        #if len(group_dict[group])<=1 or len(group_dict[group])>10:
        #continue
        print(group)
        prob = linear_chain(group,group_dict,prob_dict,label_dict,cos_dict)
        noisy_prob = noisy_or(group,group_dict,prob_dict,label_dict,cos_dict)
        print(prob,noisy_prob)
        if len(group_dict[group]) == 1:
            prob = prob_dict[group_dict[group][0]]
            noisy_prob = prob_dict[group_dict[group][0]]
        chain_outfile.write(group[0] + '\t' + group[1] + '\t' + str(prob) + '\t' + str(label_dict[group]) + '\n')
        if 1 < len(group_dict[group]) <= 10:
            prob = pair_wise(group,group_dict,prob_dict,label_dict,cos_dict)
        outfile.write(group[0] + '\t' + group[1] + '\t' + str(prob) + '\t' + str(label_dict[group]) + '\n')
        noisy_out.write(group[0] + '\t' + group[1] + '\t' + str(noisy_prob) + '\t' + str(label_dict[group]) + '\n')
        noisy_ors.append(noisy_prob)
        hybrids.append(prob)
        
        if len(group_dict[group]) == 1:
            sizes.append('n=1')
        elif len(group_dict[group]) > 10:
            sizes.append('n > 10')
        else:
            sizes.append('1<n<=10')

        labels.append(label_dict[group])
    label_dict ={1:'+',0:'.'}
    color_dict = {0:'blue',1:'red',2:'green'}
    hybrids = numpy.array(hybrids)
    noisy_ors = numpy.array(noisy_ors)
    sizes = numpy.array(sizes)
    labels= numpy.array(labels)
    fig,ax = plt.subplots()
    for g in range(len(numpy.unique(sizes))):
        g_label = numpy.unique(sizes)[g]
        ix = numpy.where(sizes==g_label)
        for l in numpy.unique(labels):
            i = numpy.where(labels[ix]==l)
            print(i)
            ax.scatter(noisy_ors[ix][i],hybrids[ix][i],label=g_label,color=color_dict[g],marker=label_dict[l])
    ax.legend()
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    plt.xlabel('noisy or')
    plt.ylabel('hybrid')
    plt.savefig('scatter.png')
    chain_outfile.close()
    outfile.close()
    noisy_out.close()

    fig,ax = plt.subplots()
    for g in range(len(numpy.unique(sizes))):
        g_label = numpy.unique(sizes)[g]
        ix = numpy.where(sizes==g_label)
        for l in numpy.unique(labels):
            i = numpy.where(labels[ix]==l)
            ax.scatter(noisy_ors[ix][i],numpy.zeros_like(noisy_ors[ix][i]),label=g_label,color=color_dict[g],marker=label_dict[l])
        ax.legend()
        plt.xlabel('noisy or')
        plt.savefig('noisy_or_line.png')

    fig,ax = plt.subplots()
    for g in range(len(numpy.unique(sizes))):
        g_label = numpy.unique(sizes)[g]
        ix = numpy.where(sizes==g_label)
        for l in numpy.unique(labels):
            i = numpy.where(labels[ix]==l)
            ax.scatter(hybrids[ix][i],numpy.zeros_like(hybrids[ix][i]),label=g_label,color=color_dict[g],marker=label_dict[l])
    ax.legend()
    plt.xlabel('hybrid prob')
    plt.savefig('hybrids_line.png')
    chain_outfile.close()
    outfile.close()
    noisy_out.close()



if __name__ == '__main__':
    main()
