# Conspiracy Theory Classifier

## Overview
The Conspiracy Theory (CT) Classifier is a comprehensive pipeline designed to analyze text documents. It classifies documents as CT-related or not, clusters them based on thematic similarities, labels each cluster, and extracts named entities.

For an in-depth understanding of the methodology and pipeline, please refer to our published paper:
- Developing a Hierarchical Model for Unraveling Conspiracy Theories

## Input
The pipeline accepts a CSV file with a dedicated column for text documents. We recommend minimal preprocessing, such as removing extra whitespace and special characters (e.g., `\n`, `\t`). Advanced preprocessing like lemmatization, punctuation removal, and stop word filtering is not required.

## Output
The pipeline generates two output files:
1. `output.csv`: Enhances the input CSV with additional columns:
   - `Label`: Binary values (0 or 1), where 1 indicates a conspiracy-related document.
   - `Cluster`: Assigns a cluster number to conspiracy-related documents.
   - `Label1`: Describes the cluster theme, based on the most frequent action-object pairs.
   - `Label2`: Provides an alternative cluster label, correlating cluster contents with predefined labels (not mentioned in the paper)

2. `entities.txt`: Lists the top 10 Named Entities from each cluster. Modify line 49 in `ner.py` to adjust the number of extracted entities.

## Dependencies
- numpy
- pandas
- pickle
- hyperopt
- sentence_transformers
- tensorflow
- transformers
- torch
- umap
- hdbscan
- functools
- spacy
- datasets


## Usage
Execute the pipeline with the following command:
	python main.py ‘your_data.csv’ ‘text_column_name’ ‘threshold_value’

- `your_data.csv`: Input CSV file with a column for text documents.
- `text_column_name`: Column name containing the text documents.
- `threshold_value`: Classifier threshold for determining CT-related documents (e.g., 0.6).

Hyperparameters can be adjusted in `main.py`, defined as hspace. The default settings offer optimal performance without excessive CPU usage. For details, see Section 4.2 of the referenced paper.

## Keyphrase Extraction
An auxiliary tool, Keyphrase Extraction, leverages HuggingFace to annotate documents with keyphrases. Although not part of the main pipeline, it aids in initial dataset labeling. The implementation is available in `keyphrase_extraction.py`.

## Version History
- 0.1: Initial Release

## Contact
- **Mohsen Ghasemizade**
- Email: mghasemizade97@gmail.com

## Citation
If you utilize this pipeline, please cite our work as follows:

**MLA:**
Ghasemizade, Mohsen, and Jeremiah Onaolapo. "Developing a hierarchical model for unraveling conspiracy theories." EPJ Data Science 13.1 (2024): 31.

**BibTeX:**

@article{ghasemizade2024developing,
  title={Developing a hierarchical model for unraveling conspiracy theories},
  author={Ghasemizade, Mohsen and Onaolapo, Jeremiah},
  journal={EPJ Data Science},
  volume={13},
  number={1},
  pages={31},
  year={2024},
  publisher={Springer Berlin Heidelberg}
}

