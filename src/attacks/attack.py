from abc import ABC
from typing import List, Tuple
import random
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import copy
import json
import spacy
import re

class Attack(ABC):
    def __init__(self):
        self.name = None

    def attack(self, sentences: List[str]) -> Tuple[str, int]:
        changes = 0
        return ' '.join(sentences), changes

    def __call__(self, sentences: List[str]) -> str:
        return self.attack(sentences)

    def __name__(self):
        return self.name


class NoAttack(Attack):
    def __init__(self):
        super().__init__()
        self.name = "NoAttack"

    def attack(self, sentences: List[str]) -> str:
        return ' '.join(sentences), 0


class ShuffleAttack(Attack):
    def __init__(self):
        super().__init__()
        self.name = "Shuffle"

    def attack(self, sentences: List[str]) -> str:
        new_sentences = copy.deepcopy(sentences)
        random.shuffle(new_sentences)

        changes = 0
        for s, s_new in zip(sentences, new_sentences):
            if s != s_new:
                changes += 1

        return ' '.join(new_sentences), changes


class BritishToAmericanEnglish(Attack):
    def __init__(self, dictionary_path):
        super().__init__()
        self.name = "BritishToAmericanEnglish"
        self.dictionary = json.load(open(dictionary_path))
        # TODO: check lemmatization

    def attack(self, sentences: List[str]) -> str:
        new_sentences = []
        changes = 0
        for sentence in sentences:
            new_sentence = []
            for word in sentence.split():
                punctuation_flag = False
                org_word = word
                # if word is arounded by puntuation, strip the puntuation but add it later
                if word.strip('.,?!') != word:
                    punctuation_flag = True
                    word = word.strip('.,?!')

                if word in self.dictionary:
                    if punctuation_flag:
                        new_sentence.append(self.dictionary[word] + org_word.strip(word))
                        changes += 1
                    else:
                        new_sentence.append(self.dictionary[word])
                        changes += 1
                else:
                    if punctuation_flag:
                        new_sentence.append(org_word)
                    else:
                        new_sentence.append(word)
                          
            new_sentences.append(" ".join(new_sentence))
        return ' '.join(new_sentences), changes


class AmericanToBritishEnglish(Attack):
    def __init__(self, dictionary_path):
        super().__init__()
        self.name = "AmericanToBritishEnglish"
        self.dictionary = json.load(open(dictionary_path))

    def attack(self, sentences: List[str]) -> str:
        new_sentences = []
        changes = 0
        for sentence in sentences:
            new_sentence = []
            for word in sentence.split():
                punctuation_flag = False
                org_word = word
                # if word is arounded by puntuation, strip the puntuation but add it later
                if word.strip('.,?!') != word:
                    punctuation_flag = True
                    word = word.strip('.,?!')

                if word in self.dictionary:
                    if punctuation_flag:
                        new_sentence.append(self.dictionary[word] + org_word.strip(word))
                        changes += 1
                    else:
                        new_sentence.append(self.dictionary[word])
                        changes += 1
                else:
                    if punctuation_flag:
                        new_sentence.append(org_word)
                    else:
                        new_sentence.append(word)

            new_sentences.append(" ".join(new_sentence))
        return ' '.join(new_sentences), changes


class NamedEntities(Attack):
    name = "NamedEntities"

    def __init__(self, model_name):
        super().__init__()
        self.name = "NamedEntities"
        ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
        ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)
        # Example ner output:
        # [{'entity': 'B-PER', 'score': 0.9990139, 'index': 4, 'word': 'Wolfgang', 'start': 11, 'end': 19},
        # {'entity': 'B-LOC', 'score': 0.999645, 'index': 9, 'word': 'Berlin', 'start': 34, 'end': 40}]

    def attack(self, sentences: List[str]) -> str:
        new_sentences = []
        entity_types = set()
        changes = 0
        for sentence in sentences:
            entities = self.ner(sentence)
            for entity in entities:
                entity_types.add(entity['entity'].split('-')[1])
            new_sentence = ""
            characters_to_pass = 0
            current_entity = None
            for i, character in enumerate(sentence):
                if i in [entity['start'] for entity in entities]:
                    current_entity = [entity for entity in entities if entity['start'] == i][0]
                    new_sentence += current_entity['entity'].split('-')[1]
                    characters_to_pass = current_entity['end'] - current_entity['start'] - 1
                elif characters_to_pass > 0:
                    characters_to_pass -= 1
                else:
                    new_sentence += character

            for entity_type in entity_types:
                # use regex
                new_sentence = re.sub(rf'({entity_type}){{2,}}', entity_type, new_sentence)

            for w, w_ in zip(sentence.split(), new_sentence.split()):
                if w != w_:
                    changes += 1

            new_sentences.append(new_sentence)
        return ' '.join(new_sentences), changes


class WordCorruption(Attack):
    def __init__(self, percent_of_words_to_corrupt: float, corrupted_word: str):
        super().__init__()
        self.name = "WordCorruption"
        self.percentage = percent_of_words_to_corrupt
        self.corrupted_word = corrupted_word

    def attack(self, sentences: List[str]) -> str:
        new_sentences = []
        changes = 0
        full_text = ' '.join(sentences)
        number_of_words_to_corrupt = int(len(full_text.split()) * self.percentage)
        left = number_of_words_to_corrupt
        for sentence in sentences:
            words = sentence.split()
            if left > 0:
                # randomly choose how many words to corrupt in this sentence
                number_of_words_to_corrupt_ = random.randint(0, left)
                left -= number_of_words_to_corrupt_
                indeces_of_words_to_corrupt = random.sample(range(len(words)), number_of_words_to_corrupt_)
            else:
                indeces_of_words_to_corrupt = []

            for index_of_word_to_corrupt in indeces_of_words_to_corrupt:
                if words[index_of_word_to_corrupt].strip('.,?!') != words[index_of_word_to_corrupt]:
                    org_word = words[index_of_word_to_corrupt]
                    # word is arounded by puntuation, strip the puntuation but add it later
                    words[index_of_word_to_corrupt] = self.corrupted_word + org_word.strip(org_word.strip('.,?!'))
                else:
                    words[index_of_word_to_corrupt] = self.corrupted_word
                changes += 1
            new_sentences.append(" ".join(words))
        return ' '.join(new_sentences), changes


class Lemmatization(Attack):
    def __init__(self):
        super().__init__()
        self.name = "Lemmatization"
        self.nlp = spacy.load("en_core_web_sm")
        self.lemmatizer = self.nlp.get_pipe("lemmatizer")

    def attack(self, sentences: List[str]) -> str:
        new_sentences = []
        changes = 0
        for sentence in sentences:
            doc = self.nlp(sentence)
            processed = self.lemmatizer(doc)
            new_sentence = " ".join([token.lemma_ for token in processed])
            # delete spaces before punctuation
            new_sentence = re.sub(r'\s([?.!"](?:\s|$))', r'\1', new_sentence)
            new_sentences.append(new_sentence)

        for w, w_ in zip(' '.join(sentences).split(), ' '.join(new_sentences).split()):
            if w != w_:
                changes += 1

        return ' '.join(new_sentences), changes


class LetterMasking(Attack):
    def __init__(self, percentage_of_letters_to_mask: float):
        super().__init__()
        self.name = "LetterMasking"
        self.masking_dictionary = {
            'a': ['@', '4'],
            'b': ['8', '6'],
            'c': ['(', '{', '[', '<'],
            'd': ['|)', '|}', '|]', '|>'],
            'e': ['3'],
            'f': ['|=', 'ph'],
            'g': ['9', '6'],
            'h': ['#'],
            'i': ['1', '!', '|'],
            'j': ['_|'],
            'k': ['|<', '|{'],
            'l': ['|', '1', '7'],
            'm': ['|v|', '|\\/|', '/\\/\\'],
            'n': ['|\\|', '/\\/'],
            'o': ['0'],
            'p': ['|2'],
            'q': ['9'],
            'r': ['|2', '|?', '|-'],
            's': ['5', '$'],
            't': ['7', '+'],
            'u': ['|_|'],
            'v': ['\\/', '|/', '\\\\//'],
            'w': ['\\/\\/', '|/\\|', '\\|/'],
            'x': ['><', '}{'],
            'y': ['`/'],
            'z': ['2', '7']
        }
        self.percentage = percentage_of_letters_to_mask

    def attack(self, sentences: List[str]) -> str:
        new_sentences = []
        changes = 0
        full_text = ' '.join(sentences)
        number_of_words_to_corrupt = int(len(full_text.split()) * self.percentage)
        left = number_of_words_to_corrupt
        for sentence in sentences:
            words = sentence.split()
            if left > 0:
                # randomly choose how many words to corrupt in this sentence
                number_of_words_to_mask = random.randint(0, left)
                left -= number_of_words_to_mask
                indeces_of_words_to_mask = random.sample(range(len(words)), number_of_words_to_mask)
            else:
                indeces_of_words_to_mask = []

            for index_of_word_to_mask in indeces_of_words_to_mask:
                word = words[index_of_word_to_mask]
                new_word = ""
                for letter in word:
                    if letter.lower() in self.masking_dictionary:
                        new_word += random.choice(self.masking_dictionary[letter.lower()])
                    else:
                        new_word += letter
                words[index_of_word_to_mask] = new_word
                changes += 1
            new_sentences.append(" ".join(words))
        return ' '.join(new_sentences), changes


class WinkyEmoji(Attack):
    def __init__(self):
        super().__init__()
        self.name = "WinkyEmoji"
        self.winky_emoji = ";)"

    def attack(self, sentences: List[str]) -> str:
        new_sentences = []
        changes = 0
        for sentence in sentences:
            if random.random() < 0.5:
                new_sentences.append(sentence.strip('.?!') + ' ' + self.winky_emoji)
                changes += 1
            else:
                new_sentences.append(sentence)
        return ' '.join(new_sentences), changes


class ExclamationMark(Attack):
    def __init__(self):
        super().__init__()
        self.name = "ExclamationMark"
        self.exclamation_mark = "!"

    def attack(self, sentences: List[str]) -> str:
        new_sentences = []
        changes = 0
        for index, sentence in enumerate(sentences):
            if index < 3 or index > len(sentences) - 3:
                new_sentences.append(sentence)
            else:
                if random.random() < 0.5:
                    # change dot to exclamation mark at the end of the sentence
                    if sentence[-1] == '.':
                        new_sentences.append(sentence[:-1] + self.exclamation_mark)
                        changes += 1
                    else:
                        new_sentences.append(sentence + self.exclamation_mark)
                        changes += 1
                else:
                    new_sentences.append(sentence)
        return ' '.join(new_sentences), changes