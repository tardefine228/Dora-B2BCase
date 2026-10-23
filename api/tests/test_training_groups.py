import random
from ml.train_reranker import TrainingPair, split_query_groups
from ml.features import normalize


def test_query_partition_keeps_positive_with_its_negative_candidates():
    pairs = [TrainingPair(q, title, code, label) for q in ["Ноутбук", "Монитор", "Насос", "Лампа"]
             for title, code, label in [("correct", "1", 1), ("wrong", "2", 0)]]
    train, validation = split_query_groups(pairs, random.Random(1))
    assert {normalize(p.query) for p in train}.isdisjoint(normalize(p.query) for p in validation)
    for partition in [train, validation]:
        for query in {p.query for p in partition}:
            assert sorted(p.label for p in partition if p.query == query) == [0, 1]
    assert split_query_groups(pairs, random.Random(1)) == (train, validation)


def test_conflicting_duplicate_label_is_not_taught_as_negative():
    pairs = [TrainingPair("Ноутбук", "computer", "1", 0),
             TrainingPair("ноутбук", "computer", "1", 1),
             TrainingPair("Монитор", "screen", "2", 1)]
    a,b = split_query_groups(pairs, random.Random(1))
    matched = [p for p in a+b if normalize(p.query) == normalize("ноутбук")]
    assert len(matched) == 1
    assert matched[0].label == 1


def test_normalized_group_does_not_split_again_by_original_case():
    pairs = [TrainingPair("Ноутбук", "computer", "1", 1),
             TrainingPair("ноутбук", "screen", "2", 0),
             TrainingPair("Монитор", "screen", "2", 1)]
    a,b = split_query_groups(pairs, random.Random(1))
    assert len({p.query for p in a+b if normalize(p.query) == normalize("ноутбук")}) == 1
