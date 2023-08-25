import string
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# Frase da processare
from pandas import read_csv

nltk.download('punkt')
nltk.download('stopwords')

out = "Dataset_LearningW.csv"
inp = "Dataset_Learning.csv"


with open(f"data/data_test/{out}","w")as file:
    file.write("Frasi;Contesto\n")

punctuation = string.punctuation
def write_file(filtered_frase,context):
    file = open(f"data/data_test/{out}","a")
    frase = ' '.join(filtered_frase)
    frase = unicodedata.normalize('NFKD', frase).encode('ascii', 'ignore').decode('ascii')
    file.write(f"{frase};{context}\n")

data = read_csv(f"data/data_test/{inp}", sep=';', encoding='utf-8')

stop_wordsIT = set(stopwords.words("italian"))
stop_wordsEN = set(stopwords.words("english"))

for index, row in data.iterrows():
    filtered_tokens = []
    tokens = word_tokenize(row["Frasi"])
    for word in tokens:
        if word.lower() not in stop_wordsIT and word.lower() not in stop_wordsEN:
            filtered_tokens.append(word)
    write_file(filtered_tokens, row["Contesto"])
    
        