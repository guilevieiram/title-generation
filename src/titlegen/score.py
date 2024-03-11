from typing import List
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'])


def score(sentence: str, reference: str) -> float:
    return scorer.score(sentence, reference)['rougeL'][2]


def avg_score(sentences: List[str], references: List[str]) -> float:
    assert len(sentences) == len(references), "Must be the same length"
    scores = [score(sent, ref) for sent, ref in zip(sentences, references)]
    return sum(scores)/len(scores)
