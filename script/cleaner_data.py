import string
import pandas as pd
import random
import string

#-------------------- Function --------------------

# Redefine the synonyms dictionary
synonyms_dict = {
    'ora': ['adesso', 'momento', 'subito'],
    'now': ['right now', 'moment', 'immediately'],
    'dire': ['affermare', 'dichiarare', 'pronunciare'],
    'say': ['state', 'declare', 'pronounce'],
    'giorno': ['giornata'],
    'day': ['date',],
    'tomorrow': ['the day after', 'the near future', 'the next day'],
    'yesterday': ['the day before', 'the recent past', 'the previous day'],
    'domani': ['il giorno dopo', 'il futuro prossimo', 'la prossima giornata'],
    'ieri': ['il giorno precedente', 'il passato recente', 'la giornata scorsa'],
    'puoi': ['potresti', 'riesci', 'sai'],
    'can': ['could', 'are able to', 'know how to'],
    'vedere': ['osservare', 'guardare', 'notare'],
    'see': ['observe', 'watch', 'notice'],
    
    'dimmi': ['dici', 'informami', 'comunicami'],
    'ultime': ['recenti', 'nuove'],
    'novità': ['aggiornamenti', 'notizie'],
    'aggiornamenti': ['novità', 'news'],
    'recenti': ['ultimi', 'nuovi'],
    'raccontami': ['dici', 'informami'],
    'notizie': ['informazioni', 'aggiornamenti'],
    'ultima': ['recente', 'finale'],
    'sapere': ['conoscere', 'informare'],
    'fatti': ['eventi', 'situazioni'],
    
    'tell': ['inform', 'notify', 'say'],
    'latest': ['most recent', 'newest'],
    'news': ['updates', 'information'],
    'recent': ['new', 'fresh'],
    'updates': ['news', 'information'],
    'stories': ['tales', 'narratives'], 
    'imposta': ['configura', 'stabilisci'],  
    'reminder': ['memo', 'alert', 'notification'],
    
    #"set": ["create", "schedule", "start", "activate", "set up"],
    "timer": ["countdown", "stopwatch", "alarm", "signal", "alert"],
    
    
    "imposta": ["metti", "alza", "abbassa", "porta", "regola", "modifica", "cambia", "aumenta", "diminuisci", "riduci", "poni", "incrementa", "rendi"],
    #"volume": ["suono", "audio"],
    
    "riproduci": ["play", "partenza", "avvia", "inizia", "metti"],
    "brano": ["pezzo", "canzone", "disco", "album", "traccia"],
    "suonare": ["riprodurre", "mettere", "attivare", "avviare"],
    "musica": ["melodia", "audio", "playlist", "selezione musicale"],
    
    
    "play": ["start", "starts", "put", "activate", "cue up"],
    "track": ["song", "album", "tune", "melody", "piece"],
    "music": ["tunes", "tracks", "songs", "melodies", "audio"],
    "playback": ["playing", "sound", "tunes", "audio", "reproduction"],
    "playlist": ["selection", "mix", "setlist", "compilation", "collection"],


    "set": ["adjust", "change", "turn", "modify", "configure"],
    "volume": ["sound", "audio", "noise", "level", "intensity"],
    "turn": ["rotate", "switch", "dial", "twist", "flip"],
    "maximum": ["highest", "peak", "top", "utmost", "max"],

}


# Caratteri di punteggiatura originali
original_punctuation = string.punctuation

# Caratteri di punteggiatura che desideri mantenere (per esempio, punto e virgola e punto esclamativo)
exceptions =  ":'"

# Crea una nuova stringa di punteggiatura rimuovendo le eccezioni
custom_punctuation = "".join([char for char in original_punctuation if char not in exceptions])

# Function to replace words with synonyms
def replace_words_with_dict(sentence):
    new_sentence = []
    for word in sentence.split():
        if word.lower() in synonyms_dict:
            new_sentence.append(random.choice(synonyms_dict[word.lower()]))
        else:
            new_sentence.append(word)
    return ' '.join(new_sentence)


# Function to pre-process text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans(' ', ' ', custom_punctuation))
    # Remove stopwords and lemmatize the words
    words = text.split()
    return ' '.join(words)


# Function to shuffle words in a sentence
def shuffle_sentence(sentence):
    words = sentence.split()
    random.shuffle(words)
    return ' '.join(words)

#-------------------- Learning cleaning --------------------
# Carica il dataset di apprendimento italiano
dataset_learning_ita = pd.read_csv("appoggio.csv", delimiter=';').drop_duplicates()

cleaned_text = dataset_learning_ita.copy()
cleaned_text['Frasi'] = cleaned_text["Frasi"].apply(preprocess_text)

shuffled_dataset = cleaned_text.copy()
shuffled_dataset['Frasi'] = shuffled_dataset['Frasi'].apply(shuffle_sentence)

# Concatenate the original and shuffled datasets to create an augmented dataset
augmented_dataset = pd.concat([cleaned_text, shuffled_dataset], ignore_index=True)

# Apply word replacement to the 'Frasi' column
replaced_dataset = augmented_dataset.copy()
replaced_dataset['Frasi'] = replaced_dataset['Frasi'].apply(replace_words_with_dict)

# Concatenate to create a final augmented dataset
final_augmented_dataset = pd.concat([augmented_dataset, replaced_dataset], ignore_index=True)
final_augmented_dataset.to_csv("appoggio.csv", sep=";", index=False, encoding='utf-8')

print(len(final_augmented_dataset))

#-------------------- Testin Cleaning --------------------


# Carica il dataset di apprendimento italiano
dataset_Testing_ita = pd.read_csv("data_it/Dataset_Testing_ita.csv", delimiter=';').drop_duplicates()
dataset_Testing_ita['Frasi'] = dataset_Testing_ita['Frasi'].apply(preprocess_text)
dataset_Testing_ita.to_csv("data_it/Dataset_Testing_ita.csv", sep=";", index=False, encoding='utf-8')

# Carica il dataset di apprendimento inglese
dataset_Testing_eng = pd.read_csv("data_eng/Dataset_Testing_eng.csv", delimiter=';').drop_duplicates()
dataset_Testing_eng['Frasi'] = dataset_Testing_eng['Frasi'].apply(preprocess_text)
dataset_Testing_eng.to_csv("data_eng/Dataset_Testing_eng.csv", sep=";", index=False, encoding='utf-8')
