import math
import fire
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Sequence, Dict, Any

MAX_NGRAM = 4


class BM25Okapi:
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.corpus = corpus
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word] += 1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        self._calc_idf(nd)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score


def extract_all_word_ngrams(
    line: str, min_order: int, max_order: int
) -> Tuple[Counter, int]:
    """Extracts all ngrams (min_order <= n <= max_order) from a sentence.
    :return: a Counter object with n-grams counts and the sequence length.
    """
    ngrams = []
    tokens = line.split()

    for n in range(min_order, max_order + 1):
        for i in range(0, len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))

    return Counter(ngrams)


def extract_reference_info(refs: Sequence[str]) -> Dict[str, Any]:
    ngrams = None
    # ref_lens = []

    for ref in refs:
        # extract n-grams for this ref
        this_ngrams = extract_all_word_ngrams(ref, 1, MAX_NGRAM)
        # ref_lens.append(ref_len)

        if ngrams is None:
            # Set it directly for first set of refs
            ngrams = this_ngrams
        else:
            # Merge counts across multiple references
            # The below loop is faster than `ngrams |= this_ngrams`
            for ngram, count in this_ngrams.items():
                ngrams[ngram] = max(ngrams[ngram], count)

    return ngrams


def compute_segment_statistics(hypothesis: str, cache_ref_ngrams: Dict) -> List[int]:
    ref_ngrams = cache_ref_ngrams
    # Extract n-grams for the hypothesis
    hyp_ngrams = extract_all_word_ngrams(hypothesis, 1, MAX_NGRAM)
    # Count the stats
    # Although counter has its internal & and | operators, this is faster
    correct = [0 for i in range(MAX_NGRAM)]
    total = correct[:]
    for hyp_ngram, hyp_count in hyp_ngrams.items():
        # n-gram order
        n = len(hyp_ngram) - 1
        # count hypothesis n-grams
        total[n] += hyp_count
        # count matched n-grams
        if hyp_ngram in ref_ngrams:
            correct[n] += min(hyp_count, ref_ngrams[hyp_ngram])

    # Return a flattened list for efficient computation
    return correct + total


def smooth_log(num: float) -> float:
    if num == 0.0:
        return -9999999999
    return math.log(num)


def compute_score(stats):
    matched = stats[:MAX_NGRAM]
    source_cnt = stats[MAX_NGRAM:]
    smooth_val = 1.0

    r_scores = [0.0 for x in range(MAX_NGRAM)]
    score = 0.0
    if not any(matched):
        return score

    for n in range(1, len(r_scores) + 1):
        if source_cnt[n - 1] == 0:
            break

        eff_order = n

        if matched[n - 1] == 0:
            smooth_val *= 2
            r_scores[n - 1] = 100.0 / (smooth_val * source_cnt[n - 1])
        else:
            r_scores[n - 1] = 100.0 * matched[n - 1] / source_cnt[n - 1]

    score = math.exp(sum([smooth_log(r) for r in r_scores[:eff_order]]) / eff_order)

    return score


def get_top_n(scores, documents, n=100):
    top_n_idx = np.argsort(scores)[::-1][:n]
    return top_n_idx, [documents[i] for i in top_n_idx]


def retrieval_from_store(data_store, idx_list):
    results = []
    for idx in idx_list:
        hit_record = data_store[idx]
        assert idx == hit_record["index"]
        try:
            hyp = hit_record["hyp"]
        except KeyError as e:
            hyp = hit_record["result"]["output"]
            print(idx)

        results.append(
            {
                "src": hit_record["src"],
                "hyp": hyp,
                "ref": hit_record["ref"],
                "op": hit_record["op"],
            }
        )
    return results


def read_json(path):
    """
    Read the json file and return a list of dictionary
    """
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def main(data_store_path, test_set_path, store_size=None):
    data_store = read_json(data_store_path)
    test_set = read_json(test_set_path)

    if store_size is not None:
        data_store = data_store[:store_size]
    print(f"Data Store{store_size}")

    tokenized_corpus = []

    for item in data_store:
        tokenized_corpus.append(item["src_token"])

    results = []
    bm25 = BM25Okapi(tokenized_corpus)
    # retrival_map = dict(zip(tokenized_corpus, list(range(len(data_store)))))

    for test_idx, query in enumerate(tqdm(test_set)):
        tokenized_query = query["src_token"]

        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_topk_idx, bm25_topk = get_top_n(bm25_scores, tokenized_corpus, 200)

        cache_ngrams = [extract_reference_info([candidate]) for candidate in bm25_topk]

        rerank_scores = []

        for candidate in cache_ngrams:
            stats = compute_segment_statistics(tokenized_query, candidate)
            score = compute_score(stats)
            rerank_scores.append(score)

        rerank_topk_idx, _ = get_top_n(rerank_scores, bm25_topk, 5)
        rerank_topk_idx = [bm25_topk_idx[idx] for idx in rerank_topk_idx]

        results.append(
            {
                "index": test_idx,
                "src": query["src"],
                "bm25_top5": retrieval_from_store(data_store, bm25_topk_idx[:5]),
                "rerank_top5": retrieval_from_store(data_store, rerank_topk_idx),
            }
        )

    with open(f"retrieval_records_{store_size}.json", "w") as outfile:
        outfile.write(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)
