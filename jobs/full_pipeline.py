import os
from jobs import Job
import gin
from src.attacks import Attack
from src.data_preparation.preprocess import clean_text, get_sentences, text_normalization
from src.metrics import MetricOriginalTextToSummary, MetricSummarytoSummary
from typing import List
import pandas as pd
from src.data_preparation.dataset_loading import dataset_loading
from src.summarization import Summarizer
import time
import openai
from scipy.stats import mannwhitneyu, ks_2samp


@gin.configurable
class FullPipeline(Job):

    def __init__(self, attacks: List[Attack], metrics_sum: List[MetricSummarytoSummary],
            metrics_org: List[MetricOriginalTextToSummary], produce_summaries: bool,
            summarizer: Summarizer, output_dir: str, stylometrix_path: str) -> None:
        self.name = 'Full Pipeline'
        self.attacks = attacks
        self.metrics_sum = metrics_sum
        self.metrics_org = metrics_org
        self.produce_summaries = produce_summaries
        self.output_dir = output_dir
        self.summarizer = summarizer
        self.stylometrix_path = stylometrix_path

    def run(self):

        print("Running attacks...")
        if self.attacks is None:
            self.attacks = []
        print(f"Attacks: {' '.join(attack.__name__ for attack in self.attacks)} ")
        if self.metrics_sum is None:
            self.metrics_sum = []
        print(f"Metrics: {' '.join(metric.__name__ for metric in self.metrics_sum)} ")
        if self.metrics_org is None:
            self.metrics_org = []
        print(f"Metrics: {' '.join(metric.__name__ for metric in self.metrics_org)} ")

        attacks = list(map(lambda x: x(), self.attacks))
        metrics_sum = list(map(lambda x: x(), self.metrics_sum))
        metrics_org = list(map(lambda x: x(), self.metrics_org))

        train, _, _ = dataset_loading()
        
        train = pd.DataFrame(train)

        print("Loading dataset...")

        df = pd.DataFrame()
        df['text'] = train['article'].apply(clean_text)
        df['summary'] = train['highlights']
        df['id'] = train['id']
        df['text'] = train['text'].apply(text_normalization)

        df['sentences'] = df['text'].apply(lambda x: get_sentences(x))

        print("Running attacks...")

        for attack_ in attacks:
            print(f"Running attack: {type(attack_).__name__}...")
            df[type(attack_).__name__] = df['sentences'].apply(lambda x: attack_(x))

        if self.produce_summaries:
            self.summarizer = self.summarizer()
            print("Producing summarization...")
            for attack_ in attacks:
                print(f"Producing summarization for attack: {type(attack_).__name__}...")
                # add waiting until rate limit resets (one minute) and retry your request
                for index, row in df.iterrows():
                    try:
                        df.loc[index, f"{type(attack_).__name__}_summary"] = self.summarizer.generate_summary(
                            text=row[f"{type(attack_).__name__}"])
                    except openai.error.RateLimitError as e:
                        print(e)
                        print("Waiting for rate limit to reset...")
                        time.sleep(60)
                        print("Retrying...")
                        df.loc[index, f"{type(attack_).__name__}_summary"] = self.summarizer.generate_summary(
                                text=row[f"{type(attack_).__name__}"])
                    except Exception as e:
                        print(e)
                        print("Couldn't generate summary...")
                        df.loc[index, f"{type(attack_).__name__}_summary"] = None
        else:
            print("Skipping summarization...")

        df.to_csv(self.output_dir)

        print("Computing metrics...")
        for metric in metrics_sum:
            print(f"Computing metric: {type(metric).__name__}...")
            for attack_ in attacks:
                df[f"{type(attack_).__name__}_{type(metric).__name__}"] = df.apply(
                    lambda x: metric.compute(x[f"{type(attack_).__name__}_summary"], x['NoAttack_summary']), axis=1)

        original_text_stylo = False
        for metric in metrics_org:
            print(f"Computing metric: {type(metric).__name__}...")
            for attack_ in attacks:
                if type(metric).__name__ == "Stylometrix":
                    if self.stylometrix_path is None:
                        raise ValueError("Stylometrix path is None!")
                    if not original_text_stylo:
                        print("Computing stylometrix for original text...")
                        original_text_stylo = True
                        stylo = metric.compute(df['text'])
                        stylo.to_csv(self.stylometrix_path + "original_text.csv")
                    stylo = metric.compute(df[f"{type(attack_).__name__}_summary"])
                    stylo.to_csv(f"{self.stylometrix_path}/{type(attack_).__name__}_summary.csv")

                else:
                    df[f"{type(attack_).__name__}_{type(metric).__name__}"] = df.apply(
                        lambda x: metric.compute(x[f"{type(attack_).__name__}"], x['text']), axis=1)

        print("Saving results...")

        # make new df only with metrics
        df_metrics = pd.DataFrame()
        for attack_ in attacks:
            for metric in metrics_sum:
                df_metrics[f"{type(attack_).__name__}_{type(metric).__name__}"] = df[f"{type(attack_).__name__}_{type(metric).__name__}"]
            for metric in metrics_org:
                if type(metric).__name__ == "Stylometrix":
                    continue
                else:
                    df_metrics[f"{type(attack_).__name__}_{type(metric).__name__}"] = df[f"{type(attack_).__name__}_{type(metric).__name__}"]

        df_metrics.to_csv(self.output_dir.replace(".csv", "_metrics.csv"))

        raport_info = []

        # analyzing results
        print("Analyzing results...")
        # we want to do statistical analysis for each stylometrix feature with respect to NoAttack
        files = os.listdir(self.stylometrix_path)
        reference_stylo = pd.read_csv(self.stylometrix_path + "/NoAttack_summary.csv")
        ommit_columns = ['text']
        columns = [column for column in reference_stylo.columns if column not in ommit_columns]
        for file in files:
            if file in ['NoAttack_summary.csv', 'original_text.csv']:
                continue
            df = pd.read_csv(self.stylometrix_path + "/" + file)
            for column in columns:
                U1, p = mannwhitneyu(reference_stylo[column], df[column])
                if p < 0.05:
                    raport_info.append(f"{file} {column} is statistically different from NoAttack (Mann-Whitney U "
                                       f"test, p={p})")

                U1, p = ks_2samp(reference_stylo[column], df[column])
                if p < 0.05:
                    raport_info.append(f"{file} {column} is statistically different from NoAttack (Kolmogorov-Smirnov "
                                       f"test, p={p})")

        # create raport
        with open(self.output_dir.replace(".csv", "_raport.txt"), "w") as f:
            f.write("Raport\n")
            f.write("======\n\n")
            f.write("Attacks:\n")
            for attack in attacks:
                f.write(f"{type(attack).__name__}\n")
            f.write("\n")       
            df['text'] = train['article'].apply(clean_text)
            df['summary'] = train['highlights']
            df['id'] = train['id']
            f.write("Metrics:\n")
            for metric in metrics_sum:
                f.write(f"{type(metric).__name__}\n")
            for metric in metrics_org:
                f.write(f"{type(metric).__name__}\n")
            f.write("\n")
            f.write("Results:\n")
            f.write("--------\n\n")
            for info in raport_info:
                f.write(info + "\n")
