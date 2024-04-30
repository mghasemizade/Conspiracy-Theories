import os
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.FIRST,
        )
        return np.unique([result.get("word").strip() for result in results])
# Load pipeline
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)

file1 = open('kbir-inspect.txt', 'a')
try:
  for dirpath, dirnames, files in sorted(os.walk('.')):
      for file in files:
        raw=open(file,'r')
        file.write(f'{file} \n')
        text = ''
        for line in raw:
          text += line +'\n'
                  
        text.replace('\n', ' ')
        keyphrases = extractor(text)
        print(keyphrases)
        file1.write(f'{keyphrases}\n')
        file1.write('--------------------------------------------\n')
  file1.close()
except:
  print('error')