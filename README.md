# prep
Replication code for paper: "HIV prevention policy: an econometric evaluation"

To reproduce the graphs and create the dataset, run the script reproduction_code/generate_results.py
To generate the regression tables, run prep_regressions.do in stata

Final datasets with PREP data also avaiable: cities.csv, regions.csv
This datasets are a merge of http://indicadores.aids.gov.br and gov.br/aids/pt-br/assuntos/prevencao-combinada/prep-profilaxia-pre-exposicao/painel-prep

They contain data on new HIV cases for every brazilian city from 2013 to 2022 and PREP data from 2018 to 2022.

Python dependencies:
numpy
pandas 2.0.1
seaborn
matplotlib
