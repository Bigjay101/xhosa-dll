"""P8.1 Masked-LM pseudo-log-likelihood, PLL-word-l2r (Member C). STUB.

Implement per handbook P8.1 and D1. Keep these signatures; tests: T8, T9.
"""
from __future__ import annotations  # Defer evaluation of type hints until they are needed.

import pandas as pd  # noqa: F401 -- Keep this existing import; F401 means "imported but unused".

import torch  # Provides tensors, GPU support, and inference utilities.
import torch.nn.functional as F  # Provides the cross-entropy loss used to calculate token NLL.

from src.segment import words  # Use the project's shared whitespace-based definition of a word.



def build_masks(word_ids: list[int | None], method: str = "l2r") -> list[list[int]]:
    """Build masking positions for word-level pseudo-log-likelihood."""
    # A word can contain several subword tokens: A1, A2, A3.
    # In l2r mode, predict A1 with [A1, A2, A3] hidden, A2 with [A2, A3]
    # hidden, and A3 with only [A3] hidden. Earlier pieces stay visible.
    # In orig mode, hide only the token being predicted.

    if method not in {"l2r", "orig"}:  # Reject unsupported masking strategies.
        raise ValueError(f"Unknown masking method: {method}")  # Include the bad value in the error.

    masks = []  # One list of positions to hide for each non-special prediction target.

    for i, word_id in enumerate(word_ids):  # i is a token position; word_id identifies its word.
        if word_id is None:  # Special tokens have no corresponding input word.
            continue  # Do not create a prediction for a special token.

        positions = [i]  # Always hide the token whose probability we want to measure.

        if method == "l2r":  # Also hide later pieces belonging to this same word.
            positions.extend(  # Append each matching position generated below.
                j  # Add this token's position to the mask.
                for j in range(i + 1, len(word_ids))  # Search positions strictly after i.
                if word_ids[j] == word_id  # Leave tokens from other words visible.
            )

        masks.append(positions)  # Preserve the same target order as the input tokens.

    return masks  # For [None, 0, 0, None], l2r returns [[1, 2], [2]].




def score_mlm(model, tokenizer, texts: list[str], method: str = "l2r", batch_size: int = 64, device: str = "cuda",) -> list[dict]:
    """Calculate word-aligned masked-LM pseudo-NLL for each text.

    NLL is negative log probability (natural logarithm): lower means the model
    assigns a higher probability to the observed token. Each token is predicted
    from its own masked copy of the text, so the sum is a pseudo-NLL, not the
    negative log probability of a single jointly generated sentence.

    batch_size counts token prediction tasks, including multiple pieces of a
    word. Each task still uses the entire text as context. Results follow the
    input text order, and word-level lists follow words(text) order.
    """

    if method not in {"l2r", "orig"}:  # Only the two implemented masking strategies are valid.
        raise ValueError(f"Unknown method: {method}")
    if batch_size < 1:  # Each inference batch must contain at least one prediction.
        raise ValueError("batch_size must be positive")
    if not tokenizer.is_fast:  # Fast tokenizers expose the token-to-word alignment we need.
        raise ValueError("A fast tokenizer is required for word_ids()")
    if tokenizer.mask_token_id is None:  # Masked inference needs a vocabulary ID for the mask token.
        raise ValueError("Tokenizer has no mask token")
    if not (hasattr(model, "roberta") and hasattr(model, "lm_head")):  # We call these model parts directly.
        raise ValueError("Expected an XLM-R/RoBERTa masked LM")

    target_device = torch.device(device)  # Convert a name such as "cpu" or "cuda" to a device object.
    if target_device.type == "cuda" and not torch.cuda.is_available():  # Fail clearly if a requested GPU is unavailable.
        raise RuntimeError("CUDA unavailable; use device='cpu' for a small check")

    model.to(target_device)  # Move the model parameters onto the requested CPU or GPU.
    model.eval()  # Disable training behavior such as dropout; this changes the supplied model's mode.
    results = []  # Collect one result dictionary per input text.

    with torch.no_grad():  # Skip gradient tracking to save memory during scoring.
        # Batch token prediction tasks within each text; a word may span several tasks.
        for text in texts:  # Finish scoring one text before moving to the next.
            ws = words(text)  # Split on whitespace; punctuation stays attached, whitespace is discarded.
            if not ws:  # Empty or whitespace-only input has no words to score.
                # Empty text, no words to score. Return a zeroed result.
                results.append({"total_nll": 0.0, "n_tokens": 0, "n_words": 0, "word_nll": [], "word_pieces": [],})
                continue  # Skip tokenization and inference for this empty text.

            # Tokenize the text, preserving word boundaries for later alignment.    
            # Treat ws as pre-split words, return PyTorch tensors, and never silently shorten input.
            enc = tokenizer(ws, is_split_into_words=True, return_tensors="pt", truncation=False,)

            word_ids = enc.word_ids(batch_index=0)  # Map each token to its index in ws; special tokens map to None.
            
            prediction_positions = [i for i, w in enumerate(word_ids) if w is not None]  # Score only word tokens.

            masks = build_masks(word_ids, method)  # Get the positions to hide for each prediction task.

            if len(prediction_positions) != len(masks):  # Every target must have exactly one masking plan.
                raise RuntimeError("Mask and prediction counts differ")

            input_ids = enc["input_ids"].to(target_device)  # Original vocabulary IDs; shape [1, sequence_length].
            attention_mask = enc["attention_mask"].to(target_device)  # Marks usable context positions, excluding padding.

            limit = tokenizer.model_max_length  # Read the tokenizer's advertised context limit.
            # Very large limits are commonly sentinel values meaning no known tokenizer limit.
            if limit < 1_000_000 and input_ids.shape[1] > limit:  # Include special tokens in the length check.
                raise ValueError("Text exceeds model context window")

            word_nll = [0.0] * len(ws)  # Accumulate the sum of token losses for each word.
            word_pieces = [0] * len(ws)  # Count how many subword predictions contribute to each word.

            for start in range(0, len(masks), batch_size):  # Advance through prediction tasks in batches.
                stop = min(start + batch_size, len(masks))  # Keep the final batch within the available tasks.
                batch_masks = masks[start:stop]  # Slice the masking plans; stop is excluded.
                positions = prediction_positions[start:stop]  # Slice the matching target token positions.
                size = len(positions)  # The last batch may be smaller than batch_size.

                masked_ids = input_ids.repeat(size, 1)  # Copy the full sequence into one row per task.

                for row, masked_positions in enumerate(batch_masks):  # Each row gets its own masking pattern.
                    masked_ids[row, masked_positions] = tokenizer.mask_token_id  # Replace those IDs with the mask ID.

                outputs = model.roberta(  # Run the encoder to obtain contextual vectors for every position.
                    input_ids=masked_ids,  # Shape: [size, sequence_length].
                    attention_mask=attention_mask.repeat(size, 1),  # Reuse the context mask for each sequence copy.
                    return_dict=True,  # Allow access through the named last_hidden_state attribute.
                )

                rows = torch.arange(size, device=target_device)  # Row indices: 0, 1, ..., size - 1.
                cols = torch.tensor(positions, device=target_device)  # Target sequence position for each row.

                selected_hidden = outputs.last_hidden_state[rows, cols]  # Paired indexing selects one vector per task.
                logits = model.lm_head(selected_hidden)  # Convert those vectors to vocabulary scores: [size, vocab_size].
                original_tokens = input_ids[0, cols]  # Retrieve the true, unmasked token IDs as prediction labels.

                losses = F.cross_entropy(  # Computes -log P(true token) using log-softmax internally.
                    logits.float(),  # Use float32 scores for the loss calculation.
                    original_tokens,  # Each label identifies the correct vocabulary entry.
                    reduction="none",  # Keep a separate loss per task instead of averaging.
                ).cpu().tolist()  # Move losses to CPU and convert them to ordinary Python numbers.

                for position, loss in zip(positions, losses):  # Pair each prediction target with its loss.
                    word_id = word_ids[position]  # Find the input word that owns this subword token.
                    word_nll[word_id] += float(loss)  # Add this piece's NLL to that word's running total.
                    word_pieces[word_id] += 1  # Record that one more piece of this word was scored.

            if any(n == 0 for n in word_pieces):  # Check that tokenization supplied a token for every word.
                raise RuntimeError("Some words received no subword tokens")

            results.append({  # Store this text's aggregate scores and word-level details.
                "total_nll": float(sum(word_nll)),  # Sum of all scored token NLLs; no length normalization.
                "n_tokens": int(sum(word_pieces)),  # Number of scored subword tokens; excludes special tokens.
                "n_words": len(ws),  # Number of whitespace-delimited input words.
                "word_nll": word_nll,  # Per-word summed NLL, in input word order.
                "word_pieces": word_pieces,  # Per-word subword counts, in the same order.
            })

    return results  # Return one dictionary per input text, preserving the input order.

