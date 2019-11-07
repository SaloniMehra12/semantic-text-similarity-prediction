#==========================================models used=========================================================
import csv
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.probability import FreqDist
import math
from nltk.corpus import wordnet
import numpy as np
from gensim.models.doc2vec import TaggedDocument
import gensim
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

#===============================================READING CSV DATASET============================================================
dataset=pd.read_csv("D:/Text_Similarity_Dataset.csv")
#print(dataset.head(2))
#print(dataset['text1'].head(2))
#print(dataset['text2'].head(2))

#===============================================SIMILARITY PREDICTING CODE================================================================
#-----------------------------------------------GETTING STOP WORDS(punctuations,conjuction,etc)-------------------------------------
stop_words = set(stopwords.words('english'))
#==============================================PROCESSING DATASET ========================================================
#input=text to be processed,list of stop_words,empty list(filtered sentence)to get main words of a text
def processing_data(para,stop_word=stop_words,filtered_sentence = []):
    word_tokens = word_tokenize(para)#break sentences into words and results in a list of words
    word_set.append(word_tokens)#getting list of total words in a  document
    filtered_sentence = [w for w in word_tokens if not w in stop_word]#includes words excluding in stop words
    return filtered_sentence#returns filtered sentence to the function

#----------------------------------------------filtering document------------------------------------------

def filtering_para(column):
    filter_paraList=[]
    for i in dataset[column]:
        sent_set.append(sent_tokenize(i))
        filtered_sentence=processing_data(i)
        filter_paraList.append(filtered_sentence)
    return filter_paraList
#=================================================COUNTING FREQUENCY OF EACH WORD IN DOCUMENT======================================
"""input=text  who's word frequency to be calculated ,total no of words in both text which are to be compared, 
a dictionary with all its words as key and initial value as 0"""

def frequency_count(text,total_words,dictionary):#gets frequency distribution of words in text
    freqd_text = FreqDist(text)
    text_count_dict=dict.fromkeys(total_words,0)# a dictionary with all its words as key and initial value as 0
    for word in text:
        text_count_dict[word]=freqd_text[word]#getting frequency of each word from function and storing in dict form
    return text_count_dict,freqd_text
#==================================================CALCULATING TERM FREQUENCY=======================================
#term frequency=how frequently a term occurs in a document =frequency of word in text/total no words in text
def term_frequency(freqd_text,text,total_words,dictionary):
    text_tf_dict=dict.fromkeys(total_words,0)# a dictionary with all its words as key and initial value as 0
    for word in text:
        text_tf_dict[word]=freqd_text[word]/len(text)#term frequency of each word
    return text_tf_dict

#==============================================CALCULATING INVERSE DOCUMENT FREQUENCY===================================
#INPUT text1 and text2 which need to be compaired ,a dictionary with all its words as key and initial value as 0
#inverse document frequency=total no of documents/no of documents with word in it
def inverse_document_frequency(para1,para2,total_words,dictionary):
    common_idf_dict=dict.fromkeys(total_words,0)#a dictionary with all its words as key and initial value as 0
    for word in common_idf_dict.keys():
        synonyms=word_synonyms(word)#for each word calculating synonyms of it
        for syn in synonyms:
            if syn in para1:#for each synonym cheching its availability in text1
                common_idf_dict[word] +=1#incrementing frequency of word occurence in both the texts
            if syn in para2:#for each synonym cheching its availability in  text 2
                common_idf_dict[word] +=1
    #print(common_idf_dict)
    for word,freq in common_idf_dict.items():
        if freq == 0:
            logarithm=1
        else:
            logarithm=math.log(2/(float(freq)))
        common_idf_dict[word]=1+logarithm #calculating idf of each word
    return  common_idf_dict
#=================================================CALCULATING SYNONYMS OF WORD=======================================

def word_synonyms(word):
    synonyms = []#list of synonyms of word
    for syn in wordnet.synsets(word):#for each synonym in set of synonyms
        for l in syn.lemmas():#for each meaningful synonym
            synonyms.append(l.name())#get that meaningful and siliar synonym
    x = np.array(synonyms)
    return (np.unique(x))#making unique list of synonyms
#==================================================CALCULATING TF-IDF=======================================================
#tf-idf=tf of each word in text * idf of each word in text
def calculated_tf_idf(para,text_tf_dict,common_idf_dict,total_words,dictionary):
    tf_idf_dict=dict.fromkeys(total_words,0) #a dictionary with all its words as key and initial value as 0
    for word in para:
        tf_idf_dict[word]=(text_tf_dict[word])*(common_idf_dict[word])#foreach word calculating tf_idf
    return  tf_idf_dict#returning list of tf-idfs of each word
#====================================================BUILDING TRAINING MODEL-1 ===================================================
#input=text1 and text2
def model(para1,para2,id):
    tagged_documents=[]#labels each input for comparision
    text1=TaggedDocument(words=para1,tags=[u'CONTENT_1'])#TaggedDocument tags the paragraphs
    tagged_documents.append(text1)
    text2=TaggedDocument(words=para2,tags=[u'CONTENT_2'])
    tagged_documents.append(text2)
    model=gensim.models.Doc2Vec(dm=0,alpha=0.25,vector_size=20,min_alpha=0.025,min_count=0)#converts large paragraphs into vector forms
    #dm=distributed memory,min-alpha =learning rate step,min_count=threshold
    model.build_vocab(tagged_documents)#passes tagged paragraphs into model


    for epoch in range(80):# epochs ranging
        if(epoch % 20==0):
            print('now training epoch %s' % epoch)
        model.train(tagged_documents,total_examples=model.corpus_count,epochs=model.epochs)#traing model on tagged document with epochs
        model.alpha -=0.002# decreasing rate of alpha by 0.002
        model.min_alpha=model.alpha # updating min alpha
    similarity=model.wv.n_similarity(para1,para2)# calculating similarity between text1 and text2 according to model
    g = float("{0:.2f}".format(similarity)) #getting value in 2 decimal place
    print("model 1 similarity =>",id,"=",(1.00-g))
    return (1.00-g)# getting final similarity vallue (as problem statemnt says that 0 means highly similar and 1 means highly dis similar)

#==========================================================BUILDING TRAINING MODEL-2 =========================================
def distance_computation_model(tf_idf_dict1,tf_idf_dict2):
    a=list(tf_idf_dict1.values())
    b=list(tf_idf_dict2.values())
    similarity=1-nltk.cluster.cosine_distance(a,b)
    g=float("{0:.2f}".format(similarity)) #getting value in 2 decimal place
    return (1-similarity)
#=================================================MAIN FUNCTION BUILDING====================================================
sent_set=[]
word_set=[]
#---------------------------------------------filtering column text1 in document and column text2 in documentwhich are to be compaired--------
filtered_para1=filtering_para('text1')
filtered_para2=filtering_para('text2')

listing1=[]# list of id and similarity predicted from model-1
listing2=[]#list of id and similarity predicted from model-2

for id in dataset['Unique_ID']:#calling text by their ids
    total_words=set(filtered_para1[id]).union(set(filtered_para2[id]))#seprating into words
    dictionary=dict.fromkeys((total_words),0)
    #---------------------------------------------------------------------------------
    text_count_dict1,freqd_text1= frequency_count(filtered_para1[id],total_words=total_words,dictionary=dictionary)#frequency of each word,frequency distribution of words in text1
    text_count_dict2,freqd_text2=frequency_count(filtered_para2[id],total_words=total_words,dictionary=dictionary)#frequency of each word,frequency distribution of words in text2
    text_tf_dict1=term_frequency(freqd_text1,filtered_para1[id],total_words=total_words,dictionary=dictionary)#maintaing term frequency of each word in text1
    text_tf_dict2=term_frequency(freqd_text2,filtered_para2[id],total_words=total_words,dictionary=dictionary)#maintaing term frequency of each word in text2
    common_idf_dict=inverse_document_frequency(filtered_para1[id],filtered_para2[id],total_words=total_words,dictionary=dictionary)#maintaing inverse document frequency of total word in text1 and text2
    tf_idf_dict1=calculated_tf_idf(filtered_para1[id],text_tf_dict1,common_idf_dict,total_words=total_words,dictionary=dictionary)#calculating tf-idf value for text1
    tf_idf_dict2=calculated_tf_idf(filtered_para2[id],text_tf_dict2,common_idf_dict,total_words=total_words,dictionary=dictionary)#calculating tf-idf value for text2
    #----------------------------------------------------------------------------------
    similarity1=model(filtered_para1[id],filtered_para2[id],id)#getting vector text trained by auto prediction and analysing pattern
    my_dict1 = {}  # stores id and similarity value from model-2
    my_dict1[id] = similarity1
    listing1.append(my_dict1)
    similarity2 = distance_computation_model(tf_idf_dict1,tf_idf_dict2) # geting cosine value of tf-idf of each word as similarity value
    my_dict2 = {}  # stores id and similarity value from model-2
    my_dict2[id] = similarity2
    listing2.append(my_dict2)
    print("model 2 similarity =>",id,"=",similarity2)
#======================================CSV FILE CREATION FOR SOLUTION==========================================
#--------------------------------------Csv consists of id and similarity corresponding to id ---------------------------------------
def csv_file_create(listing):
    dict2 = listing
    # field names
    with open('test2.csv', 'w') as f:
        for i in dict2:
            my_dict = i
            for key in my_dict.keys():
                f.write("%s,%s\n" % (key, my_dict[key]))
#-------------------------------------------calling of csv file function ----------------------------------------------------------
csv_file_create(listing1,"test.csv") # making result file from model 1
csv_file_create(listing2,"test2.csv")