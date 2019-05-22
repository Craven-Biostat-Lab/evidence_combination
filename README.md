# Evidence Combination from Relation Extraction Model

This program performs evidence combination using files from our relation_extraction model


## Getting Started
### Prerequisites

```
NumPy 1.14
```

## Running
### Creating different output files

Input must match format from our relation_extraction model. Example of which can be found in 
```
./tacred/tacred_hidden.txt
```
The column headers are:
```
PMID,Entity_1,Entity_2,Label,Probability_from_model,Cosine_Similarities,Instance_Groupings
```


This file takes the output from the relation_extraction model and groups instances using Noisy-OR and our undirected graphical model
```
python undirected_prediction_builder_labelled.py <input_file> <linear_output.txt> <undirected_output.txt> <noisy_or_output.txt> 
```



### Formatting for PR-Curves

Use the output from the previous step to get outputs in PR-Curve format.

```
python prediction_combiner_labelled.py <output_from_prediction_builder> <pr_curve_file.txt>
```

