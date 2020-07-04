mkdir -p data
gsutil cp gs://boolq/train.jsonl ./data
gsutil cp gs://boolq/dev.jsonl ./data