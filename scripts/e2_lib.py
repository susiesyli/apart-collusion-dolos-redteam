"""Batched generation and teacher-forced scoring shared by the E2 screens."""

import torch


def _left_pad(seqs, pad_id, device):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, width - len(s):] = torch.tensor(s)
        mask[i, width - len(s):] = 1
    pos = (mask.cumsum(1) - 1).clamp_min(0)
    return ids.to(device), mask.to(device), pos.to(device)


def batch_generate(model, contexts, pad_id, eos_ids, max_new_tokens, batch_size, device, log=None):
    """Greedy-continue each context (list of token ids). Returns (completion ids without EOS, ended flag)."""
    out = []
    for start in range(0, len(contexts), batch_size):
        ids, mask, _ = _left_pad(contexts[start:start + batch_size], pad_id, device)
        with torch.inference_mode():
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                 do_sample=False, eos_token_id=eos_ids, pad_token_id=pad_id)
        for row in gen[:, ids.shape[1]:].tolist():
            cut = next((i for i, t in enumerate(row) if t in eos_ids), None)
            out.append((row[:cut] if cut is not None else row, cut is not None))
        if log:
            log(min(start + batch_size, len(contexts)), len(contexts))
    return out


def batch_score(model, pairs, pad_id, batch_size, device):
    """Mean and per-token log-probs of each completion given its context: pairs = [(ctx_ids, comp_ids)]."""
    results = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        seqs = [c + x for c, x in chunk]
        ids, mask, pos = _left_pad(seqs, pad_id, device)
        m_max = max(len(x) for _, x in chunk)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos).logits[:, -(m_max + 1):-1]
            logp = torch.log_softmax(logits.float(), -1)
        for i, (_, comp) in enumerate(chunk):
            m = len(comp)
            if m == 0:
                results.append((float("nan"), []))
                continue
            tgt = torch.tensor(comp, device=device)
            lp = logp[i, m_max - m:].gather(-1, tgt[:, None])[:, 0].cpu()
            results.append((float(lp.mean()), lp.tolist()))
    return results
