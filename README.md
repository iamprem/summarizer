# summarizer - Summarize Product reviews in Amazon.com


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