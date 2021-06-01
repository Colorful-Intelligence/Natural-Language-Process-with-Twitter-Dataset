import pandas as pd

#%% Import twitter data
data = pd.read_csv(r"nlp_dataset.csv",encoding = "latin1")

data = pd.concat([data.gender,data.description],axis = 1)

data.dropna(axis = 0,inplace = True)

data.gender = [1 if each == "female" else 0 for each in data.gender]


#%% cleaning data

# regular expression RE [^a-zA-Z]
import re

first_description = data.description[4]
description = re.sub("[^a-zA-Z]"," ",first_description) # a'dan z'ye ve A'dan Z'ye kadar olan harfleri bulma geri kalanları " " ile değiştir  


description = description.lower() # büyük harften küçük harfe çervirme


#%% stopwords (irrevalent words) gereksiz kelimeler

import nltk # natural language tool kit
nltk.download('wordnet') # corpus diye bir klasöre indiriliyor
from nltk.corpus import stopwords # corpus klasöründen import ediliyor

#description = description.split()

# split() yerine tokanizer kullanabiliriz

description = nltk.word_tokenize(description) # cümleyi kelimelerine böl demek

# split kullanırsak "shouldn't" should ve not diye ikiye ayrılmaz
#%%
# gereksiz kelimeleri çıkar
description = [word for word in description if not word in set(stopwords.words("english"))]
#%%
# Lemmatization loved --> love (kelimelerin köklerini bulduk)
import nltk as nlp
lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]

description =" ".join(description)


#%%
description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    #description = [word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description] # kelimlerin köklerini bul dedik.
    description =" ".join(description)
    description_list.append(description)
    


#%% Bag of words (kelimelerin çantası)

from sklearn.feature_extraction.text import CountVectorizer # bag of words yaratmak için kullandığımız metod
max_features = 5000 # 16000 satır var bunu içinden en çok kullanılan 500 kelimeyi seç dedik
count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray() # x

print("en sık kullanılan {} kelimeler: {} ".format(max_features,count_vectorizer.get_feature_names()))

#%%
y = data.iloc[:,0].values # male or female classes
x = sparce_matrix

# train-test split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state = 42)



# %% Naive Bayes  
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train,y_train)

#%% prediction

y_pred = nb.predict(x_test)

print("Accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))

