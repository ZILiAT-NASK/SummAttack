import gin
from src.attacks import attack


def configure_class(class_object):
    return gin.external_configurable(class_object, module='attacks')


ShuffleAttack = configure_class(attack.ShuffleAttack)
BritishToAmericanEnglish = configure_class(attack.BritishToAmericanEnglish)
AmericanToBritishEnglish = configure_class(attack.AmericanToBritishEnglish)
NamedEntities = configure_class(attack.NamedEntities)
WordCorruption = configure_class(attack.WordCorruption)
Attack = configure_class(attack.Attack)
NoAttack = configure_class(attack.NoAttack)
Lemmatization = configure_class(attack.Lemmatization)
LetterMasking = configure_class(attack.LetterMasking)
WinkyEmoji = configure_class(attack.WinkyEmoji)
ExclamationMark = configure_class(attack.ExclamationMark)