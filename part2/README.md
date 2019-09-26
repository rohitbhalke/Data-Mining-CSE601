# Association Analysis

This project uses Apriori algorithm to find all the frequent itemsets from the given dataset for a given support.
Also it generates association rules based on the template provided for given support and confidence.


## Getting Started
Make sure you have Python3 installed on your computer.
You need following packages in order set up the development enviornment.
1. ***numpy***

### Installing

```buildoutcfg
pip install --upgrade pip
pip install numpy
```

### How to change configuration ?

The **config.properties** file contains the properties such as ***support*** and ***confidence*** required for the
association rules generation.

### Input Files

This project requires ***associationrulesdata.txt***, ***queries1.txt***, ***queries2.txt***,
***queries3.txt*** as input.

### How to Run

To run the association rules use following command.

```buildoutcfg
python3 TemplateQueries.py
``` 

### Code Output 
The result is generated into a file ***output.txt***. The output file contains the 
result of the queries mentioned in queries file.
The result is divided into three sections for all three queries.

The output is of following form

```
Queries for template 1
Query: RULE ANY [G59_Up]
Number of rules:  26
Query: RULE NONE [G59_Up]
Number of rules:  91


Queries for template 2
Query: RULE 3
Number of rules: 9
Query: HEAD 2
Number of rules: 6


Queries for template 3
Query: 1or1 HEAD ANY [G10_Down] BODY 1 [G59_Up]
Number of rules 24
Query: 1and1 HEAD ANY [G10_Down] BODY 1 [G59_UP]
Number of rules 0
Query: 1or2 HEAD ANY [G10_Down] BODY 2
Number of rules 11

```

## Authors

* **Pooja Shingavi** -  *pshingav*
* **Madhushri Patil** - *mvpatil*
* **Rohit Bhalke** -    *rrbhalke*
