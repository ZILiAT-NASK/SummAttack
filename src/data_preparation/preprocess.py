import re
import spacy


def delete_copywrite(text):
    """Delete copywrite from text for research purposes"""
    # E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.
    text = re.sub(r"Copyright \d{4} Reuters.", "", text)
    text = re.sub(r"All rights reserved.", "", text)
    text = re.sub(r"This material may not be published, broadcast, rewritten, or redistributed.", "", text)
    text = re.sub(r"E-mail to a friend.", "", text)
    return text


def delete_new_lines(text):
    """Delete new lines from text for research purposes"""
    text = re.sub(r"\n", "", text)
    return text


def delete_click_here_sentences(text):
    """Delete sentences with click here"""
    sentences = text.split(".")
    sentences = [sentence for sentence in sentences if "CLICK HERE" not in sentence]
    return ". ".join(sentences)


def remove_multiple_spaces(text):
    """Remove multiple spaces from text"""
    text = re.sub(r" +", " ", text)
    return text


def remove_publishing_dates(text):
    expression = r'By . (\w+ |\w+, ){1,}. (PUBLISHED: . \d+:\d+ EST, \d+ \w+ \d+ )?(. \| . )?(UPDATED: . \d+:\d+ EST, \d+ \w+ \d+ . )?'
    text = re.sub(expression, "", text)
    return text


def remove_cnn_lines(text):
    expression = r'(\w+)?( |, )?(\w+)?( )?\(CNN\)( -- )?'
    text = re.sub(expression, "", text)
    return text


def clean_text(text):
    text = delete_copywrite(text)
    text = delete_new_lines(text)
    text = delete_click_here_sentences(text)
    text = remove_multiple_spaces(text)
    text = remove_publishing_dates(text)
    text = remove_cnn_lines(text)
    return text


def get_sentences(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences


def text_normalization(text: str):
    text = re.sub(r"\n", " ", text)
    # delete multiple spaces
    text = re.sub(r" +", " ", text)
    return text