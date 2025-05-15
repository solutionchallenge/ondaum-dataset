import json

with open("./dataset/reclassified_dataset.jsonl", "r", encoding="utf-8") as fin, \
     open("./dataset/finetune_dataset.jsonl", "w", encoding="utf-8") as fout:
    for line in fin:
        try:
            obj = json.loads(line)
            out = {
                "input_text": obj["sentence"],
                "output_text": obj["emotion"]
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        except Exception:
            continue