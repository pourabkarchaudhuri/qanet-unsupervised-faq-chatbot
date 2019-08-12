from textblob import TextBlob

def spellcorrect(input):
    payload = TextBlob(input)
    spell_correction_text = payload.correct()
    return spell_correction_text

