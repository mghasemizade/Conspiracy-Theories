from imports import *
#defining the tokenizer and the ner model from huggingface
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
#callin ght ner pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

def extract_entities(text, nlp):
    ner_results = nlp(text)
    entities = []
    current_entity = []
    current_label = None
    for res in ner_results:
        word = res['word']
        label = res['entity'].split('-')[-1]
        if word.startswith('##'): #since we're using bert tokenizer, which uses mlm, it hashesh some of the letters, something we don't want in the ner output, so we're changing it back to the original word
            if current_entity:
                current_entity[-1] += word[2:]
            else:
                current_entity = [word[2:]]
        elif res['entity'].startswith('B-'): #joining the entities consisted of two words, e.g. Donald Trump
            if current_entity and len(' '.join(current_entity)) >= 3:  # Check length here
                entities.append((' '.join(current_entity), current_label))
            current_entity = [word]
            current_label = label
        elif res['entity'].startswith('I-') and label == current_label:
            current_entity.append(word)
        else:
            if current_entity and len(' '.join(current_entity)) >= 3:  # Check length here
                entities.append((' '.join(current_entity), current_label))
            current_entity = []
            current_label = None
    if current_entity and len(' '.join(current_entity)) >= 3:  # Check length here #ignoring entities less than 3 words
        entities.append((' '.join(current_entity), current_label))
    return entities

def entity_count(clusters):
    cluster_entities = {}
    overall_entities = Counter()
    for cluster_id, documents in clusters.items():
        cluster_entities[cluster_id] = Counter()
        for doc in documents:
            entities = extract_entities(doc, nlp)
            cluster_entities[cluster_id].update(entities)
            overall_entities.update(entities)
    sorted_cluster_ids = sorted(cluster_entities.keys(), key=int)  # assuming all keys can be converted to integers
    for cluster_id in sorted_cluster_ids:
        entities = cluster_entities[cluster_id]
        print(f"Cluster {cluster_id}: {entities.most_common(10)}")
        with open('entities.txt', 'a') as f:
            f.write(f"Cluster {cluster_id}: {entities.most_common(10)}")