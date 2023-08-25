from learning.learning import ML
modello = ML()

def test_1():
    result = modello.prevision("Che tempo fa")
    assert result == "MT"

def test_2():
    result = modello.prevision("Quali sono le ultime notizie?")
    assert result == "NW"

def test_3():
    result = modello.prevision("Riproduci una canzone")
    assert result == "MU"

def test_4():
    result = modello.prevision("Dimmi l'orario corrente")
    assert result == "OR"

def test_6():
    result = modello.prevision("Che giorno è oggi?")
    assert result == "GDS"

def test_7():
    result = modello.prevision("Quali eventi attuali ci sono?")
    assert result == "AL"

def test_8():
    result = modello.prevision("Chi ha vinto l'ultima partita?")
    assert result == "AL"

def test_9():
    result = modello.prevision("Qual è la capitale dell'Italia?")
    assert result == "AL"

def test_10():
    result = modello.prevision("Imposta la temperatura a 22 gradi")
    assert result == "MT"

def test_11():
    result = modello.prevision("Suona una canzone rilassante")
    assert result == "MU"

def test_12():
    result = modello.prevision("Quali film sono in programmazione al cinema?")
    assert result == "AL"

def test_13():
    result = modello.prevision("Che giorno è il prossimo compleanno di Alice?")
    assert result == "EV"

def test_14():
    result = modello.prevision("Cosa c'è nel mio calendario per domani?")
    assert result == "EV"

def test_15(): #TODO CORRECT
    result = modello.prevision("Dimmi un indovinello")
    assert result == "AL"
    
    
def test_16():
  result = modello.prevision("Prevedi pioggia per domani?") 
  assert result == "MT"

def test_17():
  result = modello.prevision("Dammi le ultime notizie di politica")
  assert result == "NW"

def test_18():
  result = modello.prevision("Riproduci Thunderstruck degli AC/DC")
  assert result == "MU"

def test_19():
  result = modello.prevision("Che ore sono adesso?")
  assert result == "OR"

def test_21():
  result = modello.prevision("Che giorno della settimana è oggi?")
  assert result == "GDS"

def test_22():
  result = modello.prevision("Fammi sapere gli ultimi aggiornamenti sulla guerra")
  assert result == "NW"

def test_23():
        result = modello.prevision("Come sarà il tempo durante il weekend?")
        assert result == "MT"

def test_24():
        result = modello.prevision("Dammi le ultime notizie di gossip")
        assert result == "NW"

def test_25():
        result = modello.prevision("Riproduci la mia playlist preferita")
        assert result == "MU"

def test_26():
        result = modello.prevision("A che ora tramonta il sole oggi?")
        assert result == "OR"

def test_28():
        result = modello.prevision("Mi dici che giorno della settimana sarà tra una settimana?")
        assert result == "GDS"
        
def test_29():
        result = modello.prevision("La gallina fa le uova")
        assert result == "AL"

def test_30():
        result = modello.prevision("L'orso vive nella foresta")
        assert result == "AL"
        
'''def test_31():
    result = modello.prevision("What's the weather like?")
    assert result == "MT"

def test_32():
    result = modello.prevision("What are the latest news?")
    assert result == "NW"

def test_33():
    result = modello.prevision("Play a song")
    assert result == "MU"

def test_34():
    result = modello.prevision("Tell me the current time")
    assert result == "OR"

def test_36():
    result = modello.prevision("What day is it today?")
    assert result == "Giorno della settimana"

def test_37():
    result = modello.prevision("What current events are there?")
    assert result == "NW"

def test_38():
    result = modello.prevision("Who won the last game?")
    assert result == "AL"

def test_39():
    result = modello.prevision("What is the capital of Italy?")
    assert result == "AL"

def test_40():
    result = modello.prevision("Set the temperature to 22 degrees")
    assert result == "MT"

def test_41():
    result = modello.prevision("Play a relaxing song")
    assert result == "MU"

def test_42():
    result = modello.prevision("What movies are showing at the cinema?")
    assert result == "AL"

def test_43():
    result = modello.prevision("When is Alice's next birthday?")
    assert result == "AL"

def test_44():
    result = modello.prevision("What's on my calendar for tomorrow?")
    assert result == "Eventi"

def test_45():
    result = modello.prevision("Tell me a riddle")
    assert result == "AL"

def test_46():
    result = modello.prevision("Will it rain tomorrow?")
    assert result == "MT"

def test_47():
    result = modello.prevision("Give me the latest political news")
    assert result == "NW"

def test_48():
    result = modello.prevision("Play Thunderstruck by AC/DC")
    assert result == "MU"

def test_49():
    result = modello.prevision("What time is it now?")
    assert result == "OR"

def test_50():
    result = modello.prevision("Turn on the kitchen light")
    assert result == "Domotica"

def test_51():
    result = modello.prevision("What day of the week is it today?")
    assert result == "Giorno della settimana"

def test_52():
    result = modello.prevision("Give me the latest updates on the war")
    assert result == "NW"

def test_53():
    result = modello.prevision("How will the weather be during the weekend?")
    assert result == "MT"

def test_54():
    result = modello.prevision("Give me the latest gossip news")
    assert result == "NW"

def test_55():
    result = modello.prevision("Play my favorite playlist")
    assert result == "MU"

def test_56():
    result = modello.prevision("What time will the sun set today?")
    assert result == "OR"

def test_57():
    result = modello.prevision("Turn off the lights in the bedroom")
    assert result == "Domotica"

def test_58():
    result = modello.prevision("Can you tell me what day of the week it will be in a week?")
    assert result == "Giorno della settimana"

def test_59():
    result = modello.prevision("The chicken lays eggs")
    assert result == "AL"
'''
def test_60():
    result = modello.prevision("Quanto manca a ferragosto")
    assert result == "MC"
    
def test_61():
    result = modello.prevision("Quanto manca al 31 dicembre")
    assert result == "MC"
    
def test_62():
    result = modello.prevision("Quanto mancano alle 12:10")
    assert result == "MC"
    
def test_63():
    result = modello.prevision("Quanto manca alle 10 meno un quarto")
    assert result == "MC"