import pandas as pd
import json
from io import StringIO

df_final = pd.DataFrame(columns=['count',
                                 'title',
                                 'pneumonia_count',
                                 'diabetes_count',
                                 'pregnant_count',
                                 'asthma_count',
                                 'copd_count',
                                 'immunosuppressant_count',
                                 'hypertension_count',
                                 'obesity_count',
                                 'ckd_count',
                                 'tobacco_count',
                                 'cardiovascular_count' ])
#word count function

#take the text and determine wordcount
pattern = ["pneumonia","diabetes", "pregnancy", "chronic obstructive pulomonary disease", "copd","asthma", "immunosuppressant", "hypertension","obesity","chronic kidney disease", "tobacco", "cardiovascular"]
pattern3_split = pattern[3].split(" ")
pattern9_split = pattern[9].split(" ")

def findWordcount(text):
    pneumonia_count = 0;
    diabetes_count = 0;
    pregnant_count = 0
    asthma_count = 0
    copd_count = 0
    immunosuppressant_count = 0
    ckd_count = 0
    hypertension_count = 0
    obesity_count = 0
    tobacco_count = 0
    cardiovascular_count = 0

    wordcount = 0

    split3_qual = 0
    split9_qual = 0

    text = text.split(" ")
    for word in text:
        word = word.lower() #convert to lowercase
        wordcount += 1
        if(pattern[0] in word): #pneumonia
            pneumonia_count += 1
        if(pattern[1] in word): #diabetes
            diabetes_count += 1
        if(pattern[2] in word): #pregnancy
            pregnant_count += 1
        if(word in pattern3_split): #check if the word is in a list
            #check if the qual is made for the previous words, and if it matches the last item
            if(split3_qual == (len(pattern3_split) - 1) and word == pattern3_split[-1]):
                copd_count += 1
                split3_qual = 0
            if(pattern3_split[split3_qual] == word): #determine if the first strings match
                split3_qual += 1
            else:
                split3_qual = 0 #reset if no match
        if(pattern[4] in word): #other way to say copd
            copd_count += 1
        if(pattern[5] in word): #asthma
            asthma_count += 1
        if(pattern[6] in word): #immunosuppressant
            immunosuppressant_count += 1
        if(pattern[7] in word): #hypertension
            hypertension_count += 1
        if(pattern[8] in word): #obesity
            obesity_count += 1
        if(word in pattern9_split):
            #print(word)
            if(split9_qual == (len(pattern9_split) - 1) and word == pattern9_split[-1]):
                ckd_count += 1
                split9_qual = 0
            if(pattern3_split[split9_qual] == word):
                split9_qual += 1
            else:
                split9_qual = 0
        if(pattern[10] in word): #hypertension
            tobacco_count += 1
        if(pattern[11] in word): #obesity
            cardiovascular_count += 1

    return [wordcount, pneumonia_count,diabetes_count,pregnant_count,asthma_count,copd_count,immunosuppressant_count,ckd_count,hypertension_count,obesity_count,tobacco_count,cardiovascular_count]


with open("dataframe.csv") as data_file:
    num = 0
    for line in data_file:
        if num==0:
            print("Header:")
            print(line)
        if(num > 0):
            text= ""

            #print(line)
            df = pd.read_csv(StringIO(line), header=None)

            #set title
            df_final.at[num, 'title'] = eval(df.at[0,4])['title']

            #append abstract text
            for item in range(len(eval(df.at[0,1]))):
                text= text+ eval(df.at[0,1])[item]['text']+ " "

            #append back_matter text
            for item in range(len(eval(df.at[0,2]))):
                text= text+ eval(df.at[0,2])[item]['text'] + " "

            #append body_text text
            for item in range(len(eval(df.at[0,3]))):
                text= text+ eval(df.at[0,3])[item]['text'] + " "

            #append title text
            text = text + eval(df.at[0,4])['title']+ " "

            #append ref_entires text
            for thing in eval(df.at[0,6]):
                text = text+ (eval(df.at[0,6])[thing]['text'])

            #call word search funtion
            liszt = findWordcount(text)

            #assign to dataframe
            df_final.at[num, 'count'] = liszt[0]
            df_final.at[num, 'pneumonia_count'] = liszt[1]
            df_final.at[num, 'diabetes_count'] = liszt[2]
            df_final.at[num, 'pregnant_count'] = liszt[3]
            df_final.at[num, 'asthma_count'] = liszt[4]
            df_final.at[num, 'copd_count'] = liszt[5]
            df_final.at[num, 'immunosuppressant_count'] = liszt[6]
            df_final.at[num, 'ckd_count'] = liszt[7]
            df_final.at[num, 'hypertension_count'] = liszt[8]
            df_final.at[num, 'obesity_count'] = liszt[9]
            df_final.at[num, 'tobacco_count'] = liszt[10]
            df_final.at[num, 'cardiovascular_count'] = liszt[11]
            #print(df_final)
        num += 1
        if (num%100==0):
            print(num)

df_final
