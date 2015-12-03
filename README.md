# Summarizer - Summarize Product Reviews

## Table of Contents
 - [Introducntion](#introduction)
 - [Dataset](#dataset)
    - [Data Collection](#data-collection)
    - [Data Preparation](#data-preparation)
 - [Methods](#methods)
    - [Latent Semantic Analysis](#latent-semantic-analysis)
    - [Text Rank](#textrank)
 - [Demos](#online-demos)
 - [Installation](#installation)
 - [Execution Instruction](#execution-instruction)
 - [Source Code](#source-code)

## Introduction
The idea of this project is to build a model for e-commerce data that summarize large amount of customer reviews of a 
product to give an overview about the product. The result of this model can be used to get an overview of what are the 
most important reveiws that many customers complaining or praising about a particular product without reading all of 
the reviews.

## Dataset
### Data Collection
The dataset used for this project is crawled from Amazon.com. The dataset contains products' reviews in separate files
for each product, each file contains maximum of 1000 reviews. Since the reviews are sorted by ranking, the first thousand
reviews are more than sufficient for the summarization task. Each review in a file contains `review_id`, `ratings`, 
`review_title`, `helpful_votes`, `total_votes` and `full_review`. The reviews file for a product is named by its 
`product_id` and all the metadata about the products are stored in a file called `iteminfo.txt`.

**Note:** Data is collected using the [customer-review-crawler](https://github.com/iamprem/customer-review-crawler) which is
**forked from [maifeng's crawler](https://github.com/maifeng/customer-review-crawler)** written in java. The original
version was outdated, so i had to rewrite the whole code that are used for data collection for this project. Please see
the commit history for the changes that I made to the forked version.

![Commit log comparison between maifen and myself(iamprem)](assets/commit_tree.png)
#### Description of dataset
    
**product_id.txt**  -- a file that contains reviews about a product.

    review_id       -   Unique id given to a review
    ratings         -   Integer value ranges from 1 to 5, describes rating of the product
    review_title    -   Punch line given by the reviewer for their review
    helpful_votes   -   Number of people found the review was helpful
    total_votes     -   Number of people upvoted or downvoted the review
    full_reveiw     -   Full review given by the reviewer

**itemsinfo.txt**   -- a file that contains all the product metadata

    product_id      -   Unique id for a product
    product_name    -   Listed name for the product in Amazon.com
    price           -   Price in US Dollars

### Data Preparation

## Dependencies

1. Python (version >2.6.6) not tested in 3.x
2. NLTK Library
3. Numpy(version >1.4)

## Installation:

### Console

    import nltk
    nltk.download('all-corpora')

### Programmatically: 
    
    python -m nltk.downloader all-corpora
    
    
### Execution Instruction

#### Summarization using LSA:

    spark-submit lsa.py -s <inputfile>

#### Summarization using TextRank:

    spark-submit textrank.py <iter-count> <summary-sent-count> <inputfile>