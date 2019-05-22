import os
import sys

import itertools


def build_shortest_dependency_path(sentence,dependency_path,dependency_matrix):
    string_path = []
    for i in range(1,len(dependency_path)):
        index_0 = dependency_path[i-1]
        index_1 = dependency_path[i]
        word_0 = sentence[index_0]
        word_1 = sentence[index_1]
        if index_0 == dependency_path[0]:
            word_0 = 'START_ENTITY'
        if index_1 == dependency_path[-1]:
            word_1 = 'END_ENTITY'
        dep_type_item = dependency_matrix[index_0][index_1]
        if '-' in dep_type_item:
            word_0,word_1=word_1,word_0
            dep_type_item = dep_type_item.replace('-','')
        string_path.append(word_0+'|'+dep_type_item+'|'+word_1)
    string_path_text = ' '.join(string_path)
    return string_path_text

def dijkstra(adj_matrix, source):
    ''' Finds shortest path between dependency paths'''
    distance = [+sys.maxint] * len(adj_matrix)  # Unknown distance function from source to v
    previous = [-1] * len(adj_matrix)  # Previous node in optimal path from source
    distance[source] = 0  # Distance from source to source
    unreached = range(len(adj_matrix))  # All nodes in the graph are unoptimized -
    
    while len(unreached) > 0:  # The main loop
        u = distance.index(min(distance))  # Get the node closest to the source
        
        if distance[u] == +sys.maxint:
            break  # all remaining vertices are inaccessible
        else:
            unreached.remove(u)
            for v in unreached:  # where v has not yet been removed from Q.
                if adj_matrix[u][v] != '':
                    alt = distance[u] + 1
                    if alt < distance[v]:  # Relax (u,v,a)
                        distance[v] = alt
                        previous[v] = u
            distance[u] = +sys.maxint  # Set the distance to u to inf so that it get's ignored in the next iteration
    return previous

def build_dependency_path_indexes(source,target,dependency_matrix):
    '''Builds and returns shortest dependency path by calling djikstras algorithm'''
    dependency_words_indexes = []
    source_token_no = source
    target_token_no = target
    previous = dijkstra(dependency_matrix, source_token_no)
    if previous[target_token_no] != -1:
        prev = previous[target_token_no]
        path = [prev, target_token_no]
        while prev != source_token_no:
            prev = previous[prev]
            path.insert(0,prev)
        dependency_words_indexes = path
        
    return dependency_words_indexes

def build_dependency_matrix(dependencies):
    dependency_matrix = [['' for y in range(len(dependencies))] for x in range(len(dependencies))]
    for i in range(len(dependencies)):
        dependency = dependencies[i]
        type = dependency[0]
        head = dependency[1]
        dependency_matrix[head][i] = type
        dependency_matrix[i][head]="-"+type
    return dependency_matrix



def main():
    pmid = '999999'
    sentence_no = 0
    
    sentence_out = open('tacred_sentences.txt','w')
    labels_out = open('tacred_labels.txt','w')
    
    with open(sys.argv[1]) as file:
        next(file)
        for line in file:
            
            #begin sentence and initialize
            if line.startswith('#'):
                sentence_no+=1
                subject = []
                object = []
                subject_type = ''
                object_type = ''
                sentence  = ['root']
                dependencies = [('ROOT',0)]
                relation = 0
                no_relation = 0
                if 'per:title' in line:
                    relation = 1
                if 'no_relation' in line:
                    no_relation = 1
                continue

            split_line = line.strip().split('\t')
            #build dependencies and paths and stuff
            if len(split_line) == 1:
                dependency_matrix = build_dependency_matrix(dependencies)
                entity_pairs = list(itertools.product(*[subject,object]))
                print(entity_pairs)
                shortest_path_length = 10000
                shortest_path = []
                for pair in entity_pairs:
                    #pair = pair[0]
                    dependency_words_indexes = build_dependency_path_indexes(pair[0],pair[1],dependency_matrix)
                    if len(dependency_words_indexes) !=0 and len(dependency_words_indexes) < shortest_path_length:
                        shortest_path_length = len(dependency_words_indexes)
                        shortest_path = dependency_words_indexes
                print(shortest_path)
                
                #build shortest word list
                shortest_path_string = build_shortest_dependency_path(sentence,shortest_path,dependency_matrix)
                print(shortest_path_string)
                
                #fix sentences
                subject_string = '_'.join(sentence[subject[0]:subject[-1]+1])
                object_string = '_'.join(sentence[object[0]:object[-1]+1])
                if subject[-1] < object[0]:
                    new_sentence = sentence[1:subject[0]]+[subject_string] + sentence[subject[-1]+1:object[0]] + [object_string] + sentence[object[-1]+1:]
                
                else:
                    new_sentence = sentence[1:object[0]]+[object_string] + sentence[object[-1]+1:subject[0]] + [subject_string] + sentence[subject[-1]+1:]

                new_sentence = ' '.join(new_sentence)
                
                
                #get locations
                subject_loc = new_sentence.find(subject_string)
                object_loc = new_sentence.find(object_string)
                
                subject_end = subject_loc + len(subject_string)
                object_end = object_loc+ len(object_string)
                
                subject_loc_pair = str(subject_loc) + ',' + str(subject_end)
                object_loc_pair = str(object_loc) + ',' + str(object_end)
                
                #build features
                if subject_type == 'PERSON' and object_type == 'TITLE':
                    label = 0
                    if relation:
                        label = 1
                    sentence_out.write(str(sentence_no) + '\t' + str(sentence_no) + '\t'+ str(subject_string) + '\t' + str(subject_loc_pair) +'\t' + str(object_string) + '\t' + str(object_loc_pair) + '\t' + str(subject_string) + '\t' + str(object_string)+ '\t' + str(subject_string) + '\t' + str(object_string) + '\tGene\tGene\t' + shortest_path_string + '\t' + new_sentence + '\n')
                    labels_out.write(str(sentence_no)+'|'+str(sentence_no)+'|'+str(subject_loc_pair)+'|'+str(object_loc_pair) + '\t'+str(label)+'\n')
                    
                
                
                continue
            
            #append sentence properties
            token_id = split_line[0]
            sentence.append(split_line[1].lower())
            if split_line[2] == 'SUBJECT':
                subject.append(int(token_id))
                subject_type = split_line[3]
            if split_line[4] == 'OBJECT':
                object.append(int(token_id))
                object_type = split_line[5]
            dependencies.append((split_line[8],int(split_line[9])))

    sentence_out.close()

if __name__ == '__main__':
    main()


