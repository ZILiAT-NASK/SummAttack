import pytest


def test_no_attack():
    from src.attacks.attack import NoAttack
    attack = NoAttack()
    attacked_sentences, changes = attack.attack(["One sentence.", "Second sentence."])
    assert attacked_sentences == "One sentence. Second sentence."
    assert changes == 0

def test_shuffle_attack():
    from src.attacks.attack import ShuffleAttack
    attack = ShuffleAttack()
    attacked_sentences, changes = attack.attack(["One sentence.", "Second sentence."])
    assert (attacked_sentences == "One sentence. Second sentence.") or (attacked_sentences == "Second sentence. One sentence.")
    assert changes == 2 or changes == 0

def test_british_to_american_english():
    from src.attacks.attack import BritishToAmericanEnglish
    attack = BritishToAmericanEnglish("data/additional_data/british_to_american.json")
    sentences = ['This is my aesthetic.', 'I do not recognise this colour.']
    attacked_sentences, changes = attack.attack(sentences)
    assert attacked_sentences == 'This is my esthetic. I do not recognize this color.'
    assert changes == 3

def test_american_to_british():
    from src.attacks.attack import AmericanToBritishEnglish
    attack = AmericanToBritishEnglish("data/additional_data/american_to_british.json")
    sentences = ['This is my esthetic.', 'I do not recognize this color.']
    attacked_sentences, changes = attack.attack(sentences)
    assert attacked_sentences == 'This is my aesthetic. I do not recognise this colour.'
    assert changes == 3

def test_named_entities():
    from src.attacks.attack import NamedEntities
    attack = NamedEntities(model_name='dslim/bert-base-NER')
    sentences = ["My name is John Doe.", "I live in New York."]
    attacked_sentences, changes = attack.attack(sentences)
    assert attacked_sentences == "My name is PER PER. I live in LOC LOC."
    assert changes == 4

def test_word_corruption():
    from src.attacks.attack import WordCorruption
    attack = WordCorruption(percent_of_words_to_corrupt=0.2, corrupted_word='MASK')
    sentences = ["My name is John Doe."]
    attacked_sentences, changes = attack.attack(sentences)
    assert (attacked_sentences == "MASK name is John Doe.") or (attacked_sentences == "My MASK is John Doe.") or (attacked_sentences == "My name MASK John Doe.") or (attacked_sentences == "My name is MASK Doe.") or (attacked_sentences == "My name is John MASK.") or (attacked_sentences == "My name is John Doe.")
    assert changes == 1 or changes == 0

def test_lemmatization():
    from src.attacks.attack import Lemmatization
    attack = Lemmatization()
    sentences = ["My name is John Doe."]
    attacked_sentences, changes = attack.attack(sentences)
    assert attacked_sentences == "my name be John Doe."
    assert changes == 2

def test_letter_masking():
    from src.attacks.attack import LetterMasking
    attack = LetterMasking(percentage_of_letters_to_mask=0.2)
    sentences = ["EAO"]
    attacked_sentences, changes = attack.attack(sentences)
    assert (attacked_sentences == "EAO") or (attacked_sentences == "3@0") or (attacked_sentences == "E4O")
    assert changes == 1 or changes == 0


def test_winky_emoji():
    from src.attacks.attack import WinkyEmoji
    attack = WinkyEmoji()
    sentences = ["This is a sentence."]
    attacked_sentences, changes = attack.attack(sentences)
    assert (attacked_sentences == "This is a sentence ;)") or (attacked_sentences == "This is a sentence.")
    assert changes == 1 or changes == 0


def test_exclamation_mark():
    from src.attacks.attack import ExclamationMark
    attack = ExclamationMark()
    sentences = ["This is a sentence."]
    attacked_sentences, changes = attack.attack(sentences)
    assert (attacked_sentences == "This is a sentence!") or (attacked_sentences == "This is a sentence.")
    assert changes == 1 or changes == 0