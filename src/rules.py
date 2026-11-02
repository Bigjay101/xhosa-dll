"""P9.2-P9.7 Rule mining, families and validation (Member B). STUB.

Rules file format: handbook D6.6.
"""
from __future__ import annotations

import pandas as pd  # noqa: F401
from collections import defaultdict, Counter
from typing import Iterable, Mapping
from difflib import SequenceMatcher
import unicodedata
import math

import numpy as np

from src.lexicon import lex_key
from src.segment import words


def edit_patterns(hist: str, cand: str) -> list[tuple]:
    """Align with difflib.SequenceMatcher; return (op, src, tgt, left_char, right_char) edits."""
    # STEP 1: Check that both inputs are strings.
    if not isinstance(hist, str) or not isinstance(cand, str):
        raise TypeError("Both spellings must be strings")

    # STEP 2: Create the sequence alignment.
    #
    # autojunk=False disables a heuristic that is unnecessary
    # for our relatively short word strings.
    matcher = SequenceMatcher(
        None,
        hist,
        cand,
        autojunk=False
    )

    edits = []

    # STEP 3: Examine the alignment operations.
    #
    # Each operation provides:
    #   op: type of operation
    #   i1:i2: affected historical characters
    #   j1:j2: corresponding modern characters
    for op, i1, i2, j1, j2 in matcher.get_opcodes():

        # Matching characters do not represent a spelling change.
        if op == "equal":
            continue

        # STEP 4: Extract the characters that changed.
        source = hist[i1:i2]
        target = cand[j1:j2]

        # STEP 5: Extract the immediate surrounding context
        # from the historical word.
        left_char = hist[i1 - 1] if i1 > 0 else ""
        right_char = hist[i2] if i2 < len(hist) else ""

        # STEP 6: Save the complete edit description.
        edits.append((
            op,
            source,
            target,
            left_char,
            right_char
        ))

    # STEP 7: Return every detected edit.
    return edits

def count_patterns(pairs: pd.DataFrame, min_support: int) -> pd.DataFrame:
    """
    Count recurring character-edit patterns in candidate spelling pairs.

    Notes
    -----
    Support counts distinct historical-modern word pairs.

    Weight sums historical token frequencies across those pairs.

    The same edit pattern occurring multiple times within one
    pair contributes only once to its support.
    """

    # STEP 1: Validate the input columns.
    required = {"hist_form", "cand_form", "hist_freq"}

    missing = required - set(pairs.columns)

    if missing:
        raise ValueError(
            f"Candidate pairs are missing columns: {sorted(missing)}"
        )

    if not isinstance(min_support, int) or min_support < 1:
        raise ValueError("min_support must be a positive integer")

    # STEP 2: Prepare counters for pattern support and weight.
    support_counts = Counter()
    weight_counts = Counter()

    # STEP 3: Ensure each historical-modern pair is counted once.
    #
    # The same pair could appear in multiple candidate categories.
    distinct_pairs = pairs.drop_duplicates(
        subset=["hist_form", "cand_form"]
    )

    # STEP 4: Process each distinct candidate pair.
    for row in distinct_pairs.itertuples(index=False):

        historical = row.hist_form
        modern = row.cand_form
        frequency = int(row.hist_freq)

        if frequency < 0:
            raise ValueError(
                "Historical frequency must be non-negative"
            )

        # Extract every character edit in this candidate pair.
        edits = edit_patterns(historical, modern)

        # STEP 5: Remove duplicate patterns within the same pair.
        # Example:
        #   tata -> thatha
        unique_edits = set(edits)

        # STEP 6: Count support and historical frequency weight.
        for pattern in unique_edits:

            support_counts[pattern] += 1
            weight_counts[pattern] += frequency

    # STEP 7: Keep patterns meeting the support threshold.
    rows = []

    for pattern, support in support_counts.items():

        if support < min_support:
            continue

        op, src, tgt, left_char, right_char = pattern

        rows.append({
            "op": op,
            "src": src,
            "tgt": tgt,
            "left_char": left_char,
            "right_char": right_char,
            "support": support,
            "weight": weight_counts[pattern],
        })

    # STEP 8: Convert the results into a DataFrame.
    columns = [
        "op",
        "src",
        "tgt",
        "left_char",
        "right_char",
        "support",
        "weight",
    ]

    result = pd.DataFrame(rows, columns=columns)

    if result.empty:
        return result

    # STEP 9: Rank the most widely supported patterns first.
    return result.sort_values(
        ["support", "weight", "op", "src", "tgt",
         "left_char", "right_char"],
        ascending=[False, False, True, True, True, True, True]
    ).reset_index(drop=True)


def _has_accented_vowel(text: str) -> bool:
    """
    Check whether a string contains a vowel with a Unicode diacritic.

    Unicode NFD decomposition separates a character such as
    'á' into the base vowel 'a' and its combining accent.
    """
    for char in text:
        decomposed = unicodedata.normalize("NFD", char)

        if (
            len(decomposed) > 1
            and decomposed[0].lower() in {"a", "e", "i", "o", "u"}
            and any(unicodedata.combining(c) for c in decomposed[1:])
        ):
            return True

    return False


def classify_pattern(op: str, src: str, tgt: str, left_char: str, right_char: str,) -> str:
    """
    P9.4: Classify a historical-to-modern edit into a rule family.

    Notes
    -----
    These labels identify candidate linguistic patterns.
    Contextual validation is performed in P9.5.
    """

    # STEP 1: Handle changes involving word boundaries.
    if op == "delete" and src == " ":
        return "WD"

    if op == "delete" and src == "-":
        return "HYP"

    # STEP 2: Identify changes involving accented vowels.
    if op in {"replace", "delete"} and _has_accented_vowel(src):
        return "DIA"

    # STEP 3: Detect insertion of h after specific consonants.
    if op == "insert" and tgt.lower() == "h":

        consonant = left_char.lower()

        if consonant in {"p", "t", "k", "c", "q", "x"}:
            return "ASP"

        if consonant == "r":
            return "VEL"

        if consonant == "b":
            return "BH"

    # STEP 4: Recognise equivalent replacements.
    if op == "replace":
        source = src.lower()
        target = tgt.lower()

        if source in {"p", "t", "k", "c", "q", "x"}:
            if target == source + "h":
                return "ASP"

        if source == "r" and target == "rh":
            return "VEL"

        if source == "b" and target == "bh":
            return "BH"

    # STEP 5: Identify selected nasal spelling changes.
    if (
        op == "insert"
        and tgt.lower() == "g"
        and left_char.lower() == "n"
        and right_char.lower() in {"c", "q", "x"}
    ):
        return "NAS"

    if (
        op == "delete"
        and src.lower() in {"n", "m"}
        and (
            left_char.lower() == src.lower()
            or right_char.lower() == src.lower()
        )
    ):
        return "NAS"

    # STEP 6: Identify selected lateral changes.
    if (
        op == "replace"
        and src.lower() == "k"
        and tgt.lower() == "t"
        and left_char.lower() == "n"
        and right_char.lower() == "l"
    ):
        return "LAT"

    if (
        op == "delete"
        and src.lower() == "h"
        and left_char.lower() == "d"
        and right_char.lower() == "l"
    ):
        return "LAT"

    # STEP 7: Detect insertion of y/w between vowels.
    vowels = {"a", "e", "i", "o", "u"}

    if (
        op == "insert"
        and tgt.lower() in {"y", "w"}
        and left_char.lower() in vowels
        and right_char.lower() in vowels
    ):
        return "SV"

    # STEP 8: Preserve unclassified patterns for investigation.
    return "OTHER"


def assign_families(patterns: pd.DataFrame) -> pd.DataFrame:
    """
    P9.4: Add a linguistic family label to each edit-pattern row.

    The input is the output of count_patterns().
    Existing support and weight values are preserved.
    """

    required = {
        "op", "src", "tgt", "left_char", "right_char",
        "support", "weight"
    }

    missing = required - set(patterns.columns)

    if missing:
        raise ValueError(f"Missing pattern columns: {sorted(missing)}")

    # Copy the table so the caller's data is not modified.
    result = patterns.copy()

    # Classify each edit using its operation and local context.
    result["family"] = result.apply(
        lambda row: classify_pattern(
            row["op"],
            row["src"],
            row["tgt"],
            row["left_char"],
            row["right_char"],
        ),
        axis=1,
    )

    return result


def classify_family(pattern: tuple) -> str:
    """Map an edit pattern to ASP, VEL, BH, NAS, LAT, SV, HYP, WD, DIA or OTHER (P9.4)."""
    raise NotImplementedError("P9.4 (Member B): implement per handbook P9.4")


def context_vectors(token_lists, targets: set, anchors: list, window: int = 2) -> dict:
    """PPMI-weighted co-occurrence vectors with anchor words within +/-window (P9.5)."""
    # STEP 1: Validate the window size.
    if window < 1:
        raise ValueError("window must be at least 1")

    # STEP 2: Normalise the target and anchor vocabularies.
    target_keys = {lex_key(word) for word in targets}
    target_keys.discard("")

    anchor_keys = [lex_key(word) for word in anchors]

    if not anchor_keys or any(not word for word in anchor_keys):
        raise ValueError("anchors must contain nonempty words")

    if len(set(anchor_keys)) != len(anchor_keys):
        raise ValueError("anchors must be unique")

    # Each anchor receives a fixed vector position.
    anchor_index = {
        word: index
        for index, word in enumerate(anchor_keys)
    }

    # STEP 3: Create sparse co-occurrence counters.
    cooccurrences = defaultdict(Counter)

    # Total co-occurrences for each target and anchor.
    target_totals = Counter()
    anchor_totals = Counter()

    total_events = 0

    # STEP 4: Process every tokenised text sequence.
    for tokens in token_lists:

        # Apply consistent word normalisation.
        normalised = [lex_key(token) for token in tokens]

        for i, word in enumerate(normalised):

            # We only build vectors for selected target words.
            if word not in target_keys:
                continue

            # Calculate the context-window boundaries.
            start = max(0, i - window)
            end = min(len(normalised), i + window + 1)

            # STEP 5: Count anchor words surrounding this target.
            for j in range(start, end):

                # A word is not its own context.
                if i == j:
                    continue

                context_word = normalised[j]

                if context_word not in anchor_index:
                    continue

                anchor_id = anchor_index[context_word]

                cooccurrences[word][anchor_id] += 1
                target_totals[word] += 1
                anchor_totals[anchor_id] += 1

                total_events += 1

    # No usable co-occurrences were observed.
    if total_events == 0:
        return {}

    # STEP 6: Convert co-occurrence counts into PPMI vectors.
    vectors = {}

    for word, counts in cooccurrences.items():

        # One feature per shared anchor word.
        vector = np.zeros(
            len(anchor_keys),
            dtype=np.float32
        )

        for anchor_id, count in counts.items():

            # PPMI = max(0, log(P(w,c)/(P(w)*P(c))))
            #
            # Using counts avoids calculating probabilities
            # separately for every target-anchor pair.
            ratio = (
                count * total_events
                / (
                    target_totals[word]
                    * anchor_totals[anchor_id]
                )
            )

            vector[anchor_id] = max(
                0.0,
                math.log(ratio)
            )

        vectors[word] = vector

    # STEP 7: Return the context vectors.
    return vectors

def select_context_anchors(
    historical_texts: Iterable[str],
    modern_lexicon: Mapping[str, int],
    candidate_pairs: pd.DataFrame,
    n_anchors: int = 2000,
    excluded_words: set[str] | None = None,
) -> list[str]:
    """
    P9.5: Select shared historical-modern context anchor words.

    Anchor selection uses historical TRAIN text and the frozen
    modern lexicon. Words involved in candidate spelling rewrites
    are excluded to reduce circularity.

    Returns the most frequent shared historical word forms,
    ordered deterministically.
    """

    # STEP 1: Validate the inputs.
    if n_anchors < 1:
        raise ValueError("n_anchors must be positive")

    required = {"hist_form", "cand_form"}

    if not required.issubset(candidate_pairs.columns):
        missing = required - set(candidate_pairs.columns)
        raise ValueError(f"Missing candidate columns: {missing}")

    # STEP 2: Identify words involved in candidate rewrites.
    #
    # Both historical and proposed modern forms are excluded.
    # For word-division candidates, split forms such as
    # "ndi yazi" into their individual components.
    rewrite_words = set()

    for column in ["hist_form", "cand_form"]:
        for form in candidate_pairs[column].dropna():

            for token in words(str(form)):
                key = lex_key(token)

                if key:
                    rewrite_words.add(key)

    # STEP 3: Count historical word frequencies.
    #
    # historical_texts MUST contain training data only.
    historical_freq = Counter()

    for text in historical_texts:
        for token in words(text):

            key = lex_key(token)

            if key and any(char.isalpha() for char in key):
                historical_freq[key] += 1

    # STEP: Normalise the optional exclusion vocabulary.
    # We use the English word list obtained during P3.
    excluded = {
        lex_key(word)
        for word in (excluded_words or set())
    }

    # STEP 4: Find identical historical-modern word forms.
    #
    # An eligible anchor must:
    # 1. Occur in historical training text.
    # 2. Exist in the frozen modern lexicon.
    # 3. Not participate in a candidate rewrite.
    # Select stable shared reference words.
    eligible = [
        word
        for word in historical_freq
        if word in modern_lexicon

        # The word cannot participate in a candidate rewrite.
        and word not in rewrite_words

        # Exclude words appearing in our English vocabulary list.
        and word not in excluded

        # Exclude single-letter and very short fragments.
        and len(word) >= 3

        # Avoid punctuation-containing forms as context anchors.
        and word.isalpha()
    ]

    # STEP 5: Rank by historical frequency.
    #
    # Modern frequency and alphabetical order break ties,
    # ensuring deterministic output.
    eligible.sort(
        key=lambda word: (
            -historical_freq[word],
            -modern_lexicon[word],
            word,
        )
    )

    # STEP 6: Enforce the required number of anchors.
    #
    # Do not silently relax our methodology if the
    # exclusion criteria leave insufficient words.
    if len(eligible) < n_anchors:
        raise ValueError(
            f"Only {len(eligible)} eligible shared anchors; "
            f"need {n_anchors}"
        )

    # STEP 7: Return the selected anchor vocabulary.
    return eligible[:n_anchors]

def family_validity(pairs: pd.DataFrame, hist_vecs: dict, mod_vecs: dict,mod_freqs: Mapping[str, int], n_boot: int = 2000, seed: int = 13, min_support: int = 20,) -> pd.DataFrame:
    """
    P9.5: Validate spelling families using contextual similarity.

    For each historical-modern candidate pair:
        1. Calculate cosine similarity between its context vectors.
        2. Select a frequency- and length-matched modern control.
        3. Calculate the candidate's similarity advantage.

    Bootstrap the advantages over distinct candidate pairs.

    A family is accepted only when:
        - At least min_support usable candidate pairs exist.
        - The lower 95% confidence bound is greater than zero.

    This implementation uses the handbook's permitted
    same-length alternative for control matching.
    """

    # STEP 1: Check the candidate table.
    required = {"hist_form", "cand_form", "family"}

    if not required.issubset(pairs.columns):
        raise ValueError(
            f"Missing columns: {required - set(pairs.columns)}"
        )

    if n_boot < 1 or min_support < 1:
        raise ValueError("Bootstrap count and support must be positive")

    rng = np.random.default_rng(seed)

    # STEP 2: Group modern words by length.
    #
    # Only words with usable vectors and positive frequencies
    # can serve as controls.
    controls_by_length = defaultdict(list)

    for word in sorted(mod_vecs):

        vector = np.asarray(mod_vecs[word], dtype=float)

        if (
            mod_freqs.get(word, 0) > 0
            and np.linalg.norm(vector) > 0
        ):
            controls_by_length[len(word)].append(word)

    # STEP 3: Prevent a historical word's proposed modern
    # candidates from serving as its negative controls.
    candidate_sets = (
        pairs.groupby("hist_form")["cand_form"]
        .agg(set)
        .to_dict()
    )

    # Count each historical-modern-family combination once.
    distinct = pairs.drop_duplicates(
        ["family", "hist_form", "cand_form"]
    ).sort_values(
        ["family", "hist_form", "cand_form"]
    )

    results = []

    # STEP 4: Evaluate each family independently.
    for family, group in distinct.groupby("family", sort=True):

        deltas = []
        n_with_vectors = 0

        for row in group.itertuples(index=False):

            historical = str(row.hist_form)
            modern = str(row.cand_form)

            historical_vector = hist_vecs.get(historical)
            modern_vector = mod_vecs.get(modern)

            # Skip pairs without usable context information.
            if historical_vector is None or modern_vector is None:
                continue

            h = np.asarray(historical_vector, dtype=float)
            m = np.asarray(modern_vector, dtype=float)

            h_norm = np.linalg.norm(h)
            m_norm = np.linalg.norm(m)

            if (
                h_norm == 0
                or m_norm == 0
                or h.shape != m.shape
            ):
                continue

            n_with_vectors += 1

            # STEP 5: Find same-length modern controls
            # with frequencies within +/-50% of the candidate.
            frequency = int(mod_freqs.get(modern, 0))

            if frequency <= 0:
                continue

            possible_controls = [
                word
                for word in controls_by_length[len(modern)]
                if word not in candidate_sets[historical]
                and 0.5 * frequency <= mod_freqs[word] <= 1.5 * frequency
            ]

            if not possible_controls:
                continue

            # Select one control reproducibly.
            control = possible_controls[
                int(rng.integers(len(possible_controls)))
            ]

            c = np.asarray(mod_vecs[control], dtype=float)

            # STEP 6: Calculate cosine similarities.
            candidate_similarity = float(
                np.dot(h, m) / (h_norm * m_norm)
            )

            control_similarity = float(
                np.dot(h, c) / (h_norm * np.linalg.norm(c))
            )

            # Positive delta means the proposed correspondence
            # has greater contextual similarity than its control.
            deltas.append(
                candidate_similarity - control_similarity
            )

        # STEP 7: Bootstrap the pair-level similarity advantages.
        n_valid = len(deltas)

        if n_valid:

            values = np.asarray(deltas, dtype=float)

            bootstrap_means = np.empty(n_boot)

            for i in range(n_boot):
                bootstrap_means[i] = rng.choice(
                    values,
                    size=n_valid,
                    replace=True,
                ).mean()

            mean_delta = float(values.mean())

            lower, upper = np.quantile(
                bootstrap_means,
                [0.025, 0.975],
            )

        else:
            mean_delta = np.nan
            lower = np.nan
            upper = np.nan

        # STEP 8: Apply the handbook's acceptance rule.
        accepted = (
            n_valid >= min_support
            and lower > 0
        )

        results.append({
            "family": family,
            "n_candidate_pairs": len(group),
            "n_with_vectors": n_with_vectors,
            "n_validated_pairs": n_valid,
            "delta": mean_delta,
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "accepted": bool(accepted),
        })

    return pd.DataFrame(results)


def load_rules(path: str) -> dict:
    """Read configs/rules_v1.yaml (handbook D6.6)."""
    raise NotImplementedError("P9.7 (Member B): implement per handbook P9.7")
