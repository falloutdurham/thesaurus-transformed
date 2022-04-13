import csv
import hnswlib
import os
import pickle
import spacy
import time

from sentence_transformers import SentenceTransformer, util
from tokenize import String
from typing import Dict


SPACY_LEMMA = spacy.load("en_core_web_sm")
MODEL_NAME = "all-mpnet-base-v2"

MODEL = SentenceTransformer(MODEL_NAME)

DATASET_PATH = "words_alpha.txt"
EMBEDDING_CACHE_PATH = f"embeddings-{DATASET_PATH}.pkl"
THESAURUS_ENTRIES_PATH = "Thesaurus-Synonyms-Common.txt"
INDEX_PATH = "./hnswlib.index"

EMBEDDING_SIZE = 768
TOP_K = 25


def create_reference_embeddings():
    if not os.path.exists(EMBEDDING_CACHE_PATH):
        corpus_sentences = set()
        with open(DATASET_PATH, encoding="utf8") as text_in:
            for line in text_in:
                corpus_sentences.add(SPACY_LEMMA(line.strip())[0].lemma_)

        corpus_sentences = list(corpus_sentences)
        print(f"Creating embeddings for {EMBEDDING_CACHE_PATH}")
        corpus_embeddings = MODEL.encode(
            corpus_sentences, show_progress_bar=True, convert_to_numpy=True
        )

        print("Dumping embedding pickle")
        with open(EMBEDDING_CACHE_PATH, "wb") as embeddings_out:
            pickle.dump(
                {"sentences": corpus_sentences, "embeddings": corpus_embeddings},
                embeddings_out,
            )
    else:
        print("Loading cached embeddings")
        with open(EMBEDDING_CACHE_PATH, "rb") as pickled:
            cache_data = pickle.load(pickled)
            corpus_sentences = cache_data["sentences"]
            corpus_embeddings = cache_data["embeddings"]
    return corpus_sentences, corpus_embeddings


def create_index(corpus_embeddings):
    index = hnswlib.Index(space="cosine", dim=EMBEDDING_SIZE)

    if os.path.exists(INDEX_PATH):
        index.load_index(INDEX_PATH)
    else:
        index.init_index(max_elements=len(corpus_embeddings), ef_construction=400, M=64)

        index.add_items(corpus_embeddings, list(range(len(corpus_embeddings))))

        print(f"Saving index to {INDEX_PATH}")
        index.set_ef(50)
        index.save_index(INDEX_PATH)

    return index


def format_markdown(entries: Dict):
    sorted_entries = sorted(entries.keys())

    def format_entry(s: String):
        if s in entries:
            return f"**{s}**"
        else:
            return s

    print("::: columns\n\n")

    for entry in sorted_entries:
        entry_synonyms = entries[entry]
        formatted = [format_entry(x) for x in entry_synonyms]
        print(f"**{entry}** — { ', '.join(formatted) } \n")

    print(":::\n\n")


def build_thesaurus(index, corpus_sentences, thesaurus_words):
    thesaurus_entries = {}

    for entry in thesaurus_words:
        question_embedding = MODEL.encode(entry)

        corpus_ids, distances = index.knn_query(question_embedding, k=TOP_K)

        hits = [
            {"corpus_id": id, "score": 1 - score}
            for id, score in zip(corpus_ids[0], distances[0])
        ]
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)

        thesaurus_entries[entry] = [
            corpus_sentences[hit["corpus_id"]] for hit in hits[1:TOP_K]
        ]

    return thesaurus_entries


def get_words():
    with open(THESAURUS_ENTRIES_PATH, encoding="utf8") as text_in:
        thesaurus_words = set()

        for line in text_in:
            word = SPACY_LEMMA((line.split("|")[0]).strip())[0].lemma_
            if len(word) >= 4:
                thesaurus_words.add(word)

    thesaurus_words = list(thesaurus_words)
    return thesaurus_words


if __name__ == "__main__":
    corpus_sentences, corpus_embeddings = create_reference_embeddings()
    index = create_index(corpus_embeddings)
    thesaurus_words = get_words()
    thesaurus_entries = build_thesaurus(index, corpus_sentences, thesaurus_words)
    format_markdown(thesaurus_entries)
