from carl.experiments.experiment import Experiment

NESTED = '__NESTED__'

base_config = {
    'job': '@jobs.FullPipeline',
    'jobs.FullPipeline.attacks': '[@attacks.NoAttack, @attacks.ShuffleAttack, @attacks.BritishToAmericanEnglish, '
                                 '@attacks.AmericanToBritishEnglish, @attacks.NamedEntities, @attacks.WordCorruption, '
                                 '@attacks.ExclamationMark, @attacks.Lemmatization, @attacks.LetterMasking, '
                                 '@attacks.WinkyEmoji]',
    'BritishToAmericanEnglish.dictionary_path': "'data/additional_data/british_to_american.json'",
    'AmericanToBritishEnglish.dictionary_path': "'data/additional_data/american_to_british.json'",
    'NamedEntities.model_name': "'dslim/bert-base-NER'",
    'WordCorruption.percent_of_words_to_corrupt': 0.2,
    'WordCorruption.corrupted_word': "'blank'",
    'LetterMasking.percentage_of_letters_to_mask': 0.2,
    'NovelNGrams.n': '[3, 4, 5]',

    # Metrics params
    'jobs.FullPipeline.metrics_sum': '[@metrics.BERTScore, @metrics.SentimentScore, @metrics.NamedEntitiesScore]',
    'jobs.FullPipeline.metrics_org': '[@metrics.Stylometrix, @metrics.NovelNGrams]',
    'jobs.FullPipeline.produce_summaries': True,
    'jobs.FullPipeline.output_dir': "'data/summaries_with_metrics_chat.csv'",
    'jobs.FullPipeline.summarizer': "@summarizers.HFSummarizer",
    'HFSummarizer.prompt': "'Generate a summary of the following text: '",
    'HFSummarizer.temperature': 0.7,
    'HFSummarizer.model_name': "'t5-base'"
}