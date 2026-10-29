"""
Builds the held-out workloads for RQ2 as token IDs, so the replay sends exactly what was built
and the prompt lengths are known before anything runs. Nothing here was used to fit the
accounting models; those were fitted on the synthetic sweep only.

  sharegpt  real multi-turn ShareGPT conversations. Turn k's prompt is the whole conversation
            so far, so later turns can reuse earlier ones. Conversations are interleaved.
  agent     a few long tool-catalog system prompts (function definitions from the Glaive
            function-calling set), each shared by many different user requests.
  rag       LongBench documents with a question after them. Popular documents come up again
            (Zipf), so their prefix gets reused. Questions are drawn from the dataset's
            question pool, not only the document's own, so repeats still differ in the suffix.

Output lengths come from the dataset's own responses, clipped, and the replay enforces them
with ignore_eos. Each file holds several windows of --requests requests.

Example: python make_workloads.py --model Qwen/Qwen2.5-7B-Instruct --out-dir workloads
"""
import argparse
import json
import os
import random
import zipfile

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MAX_PROMPT = 8192    # same top length as the synthetic sweep, so nothing is extrapolated far


def zipf_choice(rng, n):
    # Rank i is picked with weight 1/(i+1): a few items are popular, most are rare.
    return rng.choices(range(n), weights=[1 / (i + 1) for i in range(n)])[0]


class Builder:
    def __init__(self, model):
        self.tok = AutoTokenizer.from_pretrained(model)

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def chat(self, messages):
        # Rendered to text first and tokenized once, so requests with the same leading
        # messages get the same leading token IDs, which is what the radix cache matches on.
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.encode(text)

    def out_len(self, text, lo, hi):
        return max(lo, min(hi, len(self.encode(text))))


def sharegpt(b, rng, windows, n):
    path = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered",
                           "ShareGPT_V3_unfiltered_cleaned_split.json", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    convs = []
    for c in data:
        turns = c["conversations"]
        roles = [t["from"] for t in turns]
        # Only clean human/gpt alternation with at least two exchanges, so there is something to reuse.
        if len(turns) >= 4 and all(r == ("human" if i % 2 == 0 else "gpt") for i, r in enumerate(roles)):
            convs.append([t["value"] for t in turns])
    rng.shuffle(convs)

    out = []
    for w in range(windows):
        active, reqs = [], []
        while len(reqs) < n:
            if len(active) < 8:
                active.append([convs.pop(), 0])   # [conversation, next turn]
            conv = rng.choice(active)
            msgs, k = conv[0], conv[1]
            history = [{"role": "user" if i % 2 == 0 else "assistant", "content": m}
                       for i, m in enumerate(msgs[:2 * k + 1])]
            ids = b.chat(history)
            if len(ids) > MAX_PROMPT:
                active.remove(conv)
                continue
            reqs.append({"window": w, "input_ids": ids, "out": b.out_len(msgs[2 * k + 1], 16, 512)})
            conv[1] += 1
            if 2 * conv[1] + 1 >= len(msgs):
                active.remove(conv)
        out += reqs
    return out


def agent(b, rng, windows, n, n_prompts=4, prompt_tokens=3000):
    path = hf_hub_download("glaiveai/glaive-function-calling-v2", "glaive-function-calling-v2.json",
                           repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    defs, queries = [], []
    for r in data:
        system, chat = r.get("system", ""), r.get("chat", "")
        if "-\n" in system:
            defs.append(system.split("-\n", 1)[1].strip())
        if chat.startswith("USER:") and "ASSISTANT:" in chat:
            user, rest = chat[5:].split("ASSISTANT:", 1)
            queries.append((user.strip(), rest.split("<|endoftext|>")[0].strip()))
    defs = sorted(set(defs))
    rng.shuffle(defs)
    rng.shuffle(queries)

    header = ("You are an agent that can call tools. Call a tool when it helps with the user's "
              "request, otherwise answer directly. Available tools:\n\n")
    prompts = []
    for _ in range(n_prompts):
        text = header
        while len(b.encode(text)) < prompt_tokens:
            text += defs.pop() + "\n\n"
        prompts.append(text)

    out = []
    for w in range(windows):
        for _ in range(n):
            user, reply = queries.pop()
            system = prompts[zipf_choice(rng, n_prompts)]
            ids = b.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
            out.append({"window": w, "input_ids": ids, "out": b.out_len(reply, 16, 256)})
    return out


def rag(b, rng, windows, n, n_docs=20, doc_tokens=6000):
    path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    items = []
    with zipfile.ZipFile(path) as z:
        for name in ["multifieldqa_en", "qasper", "hotpotqa"]:
            with z.open(f"data/{name}.jsonl") as f:
                items += [json.loads(line) for line in f]
    rng.shuffle(items)
    questions = [(it["input"], it["answers"][0] if it["answers"] else "") for it in items]

    out = []
    for w in range(windows):
        docs = []
        for _ in range(n_docs):
            ids = b.encode(items.pop()["context"])[:doc_tokens]
            docs.append(b.tok.decode(ids))
        for _ in range(n):
            doc = docs[zipf_choice(rng, n_docs)]
            q, answer = rng.choice(questions)
            msg = f"Answer the question using the document below.\n\nDocument:\n{doc}\n\nQuestion: {q}"
            out.append({"window": w, "input_ids": b.chat([{"role": "user", "content": msg}]),
                        "out": b.out_len(answer, 16, 128)})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out-dir", default="workloads")
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--requests", type=int, default=120, help="requests per window")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    b = Builder(args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, build in [("sharegpt", sharegpt), ("agent", agent), ("rag", rag)]:
        reqs = build(b, random.Random(args.seed), args.windows, args.requests)
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w") as f:
            for r in reqs:
                f.write(json.dumps(r) + "\n")
        lens = [len(r["input_ids"]) for r in reqs]
        print(f"{name}: {len(reqs)} requests, prompt {min(lens)}-{max(lens)} "
              f"(mean {sum(lens) / len(lens):.0f}) tokens, "
              f"mean out {sum(r['out'] for r in reqs) / len(reqs):.0f}")


if __name__ == "__main__":
    main()
