# Movie Review Sentiment Analysis 🎬📊📝

1. Dataset (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews):
The dataset to be used is the "IMDB Dataset of 50K Movie Reviews," which consists of 50,000 movie reviews from IMDB for natural language processing or text analysis. This dataset is designed for binary sentiment classification and contains two columns: "review" and "sentiment." It provides a set of 25,000 highly polar movie reviews for training and 25,000 for testing (Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts, 2011). This dataset is suitable for predicting the number of positive and negative reviews using classification or deep learning algorithms.

# 2. Issues for Analysis:
<h2> 2.1 Counting the Number of Word Types Used in Reviews (Adjectives, Verbs, Connectors, etc.): </h2>
The goal is to analyze the linguistic content of movie reviews and determine how many adjectives, verbs, connectors, and other word types are used in the dataset. This will provide information about the variety of vocabulary and linguistic structures present in the reviews.

# <h2> 2.2 Finding the Most Used Words in the Reviews: </h2>
This analysis aims to identify the most frequent words in all movie reviews. Doing so can provide insights into the most commonly mentioned keywords or common themes in the opinions of the movies

# <h2> 2.3 Distribution of Word Frequencies in a Review: </h2>
The objective of this task is to examine how word frequencies are distributed in a specific review. This can provide information about which words are more repetitive or prominent in a particular review.

# <h2> 2.4 Counting the Number of Positive and Negative Reviews: </h2>
The goal here is to search for and determine how many movie reviews are considered positive and how many are negative. This will help understand the ratio of positive and negative opinions in the dataset and gain a general idea of the polarity of the reviews.

# <h2> 2.5 Finding the Most Used Adjectives in Reviews: </h2>
The aim is to identify the most common adjectives used in movie reviews. This will help understand what kind of descriptions or qualifiers are most frequently highlighted in the opinions.

# <h2> 2.6 Getting All Adjectives from Positive and Negative Reviews: </h2>
The objective here is to identify the most used adjectives in positive and negative reviews separately. Comparing adjectives in each type of review can help understand which specific aspects are more appreciated or criticized by viewers.

# <h2> 2.7 Calculating the Length of Reviews (Number of Words): </h2>
The goal here is to analyze the number of words in each review. This will provide insight into the average length of opinions, which can be useful for understanding how much detail is provided in the reviews.

# 4. Library Loading:
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/6daf9a12-ae56-4226-9a68-c3f631783701)

# 5. Data Loading and Cleaning:
First, we use the pandas library to load the dataset and store it in the df variable. Once we have the DataFrame loaded, we proceed to clean the data. In this case, we identify line breaks in the text columns and replace them with spaces. This is important to improve text consistency and facilitate further processing.
 ![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/1cdde419-6a28-4476-ac3c-e5adf42b6209)

# 6. Text Sentence Splitting
Now, we implement the task of splitting text into sentences using the spaCy library. The goal of this task is to take the cleaned reviews stored in the DataFrame and split them into sentences for a more detailed language analysis.
We start the task by importing the spaCy library and loading the 'en_core_web_sm' model. We define a function called spacy_sentence_tokenize(text). This function takes text as input and uses the loaded spaCy model to split the text into sentences. Each resulting sentence is stored in a list.
Once the sentence tokenization function is defined, it is applied to the 'cleaned_review' column of the DataFrame df, which contains the clean movie reviews. The function processes each review and returns a list of sentences corresponding to each review.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/4a664093-f938-4cc5-814a-181a885d3c8f)

# 7. Text Word Labeling
We define a function called spacy_pos_tagging(text) that takes text as input and uses the spaCy model previously loaded to label each word with its part of speech (POS).
The spacy_pos_tagging function is applied to the 'cleaned_review' column of the DataFrame df, which contains the clean movie reviews. The function processes each review and labels each word with its corresponding POS tag, generating a list of tuples for each review.
Next, a new DataFrame called tagged_words_df is created to store the labeled words along with their POS tags. The explode() function is used to convert the list of tuples into individual columns, and then the words and tags are stored in separate columns called 'word' and 'tag'.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/e1c34560-b8a4-4558-800f-d946c1f12477)

# 8. Number of Word Types Used in Reviews (Adjectives, Verbs, Connectors, etc.)
We load the 'en_core_web_sm' spaCy model and define the spacy_pos_tagging(text) function to label each word in the text with its part of speech (POS).
A new DataFrame called tagged_words_df is created to store the labeled words and their POS tags. The explode() function is used to convert the list of tuples into individual columns.
The labeled words are classified by type (adjective, noun, etc.), and the count of words of each type is calculated using the value_counts() function.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/a70a085a-f59d-45a1-8dea-7c4a90fdff1e)

#<h2>8.1 Number of Words (Graph)</h2>
The bar chart allows us to have an overview of the grammatical categories present in the reviews to understand how people who wrote the reviews use different types of words to express their opinions about the movies.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/a659d288-6786-4a79-988d-52f541e4a3ee)

# 9. Most Used Words in Reviews
We create a list called all_words to store all the words present in the reviews.
Using the FreqDist function from NLTK, we calculate the frequency of each word in the all_words list. FreqDist generates a dictionary that assigns each word to its frequency in the text.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/28e97bdb-66d1-4764-8961-595cd9eb0584)

# <h2>9.1 Word Frequency (Graph)</h2>
Thanks to the following graph, we can observe the most frequent words in the reviews, which can be useful for understanding language patterns and common themes that people mention most frequently.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/b3daaba6-97cf-4361-bfee-bdd05572c967)

# 10. Text Summary
The summary refers to a brief excerpt or abbreviated version of the content of the movie reviews.
We create an auxiliary function called get_luhn_summary(text, sentences_count=2), which takes text as input and uses the LuhnSummarizer algorithm to generate a summary of the text. The sentences_count parameter indicates the number of sentences desired in the summary and is set to 2 by default.
The get_luhn_summary function is applied to the 'cleaned_review' column of the DataFrame df, which contains the clean movie reviews, in order to create a summary for each review and store it in a new column called 'text_summary'.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/952f867a-b005-46dc-8b9f-734e0e60083f)

# 11. Word Frequency Distribution
Using the word_tokenize function from NLTK, each cleaned review in the DataFrame df is tokenized. Then, the FreqDist function is applied to the resulting list of tokens to calculate the frequency of each word.
This allows us to have a view of word frequency distributions in the movie reviews. This information can be useful for understanding which words are used most frequently in the reviews and how words are distributed in each of them.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/1098cfb8-75e5-4ab5-a162-03db0ce75d19)

# 12. Count of "positive" and "negative"
Using the value_counts() function from pandas, we count the occurrences of each label in the 'sentiment' column to obtain the number of positive and negative reviews.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/bb4552fa-223a-4b5a-800a-80a17964a1d1)

# <h2>12.1 Count of Positives and Negatives (Graph)</h2>
Through the following graph, we can observe the number of positive and negative reviews. This can be useful to understand the overall audience reaction to the movies and perform sentiment analysis or opinion evaluations.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/02bc0b76-e046-4ae2-95eb-c65885014308)

# 13. Number of Words in Each Review
Now, we calculate the number of words in each movie review and create a scatter plot to visualize the distribution of the number of words in all reviews.
Using the apply() function from pandas along with len(x.split()), we calculate the number of words in each review. The scatter() function from matplotlib is used to create a scatter plot.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/77e7d586-88a5-4f60-8845-9e551ca98b48)

# <h2>13.1 Number of Words in Reviews</h2>
The following scatter plot provides a view of the distribution of the number of words in movie reviews, which can be useful for understanding the average length of reviews and the variability in lengths.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/1988fe22-839c-447b-a6f3-a2cdbc00b6df)

# 14. Most Repeated Adjectives in Positive Reviews
Now, we perform an analysis of adjectives present in positive reviews in the dataset. First, we select the reviews from the DataFrame df that have the 'positive' label in the 'sentiment' column and store them in the variable positive_reviews.
Using the get_adjectives(text) function, we obtain the adjectives from a review. First, the text is tokenized into individual words using word_tokenize(), and only words with alphabetical characters are filtered using a regular expression.
A FreqDist function is created to calculate the frequency of each adjective in the positive_adjectives list, creating a FreqDist object that stores the frequency of each adjective.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/def4fdd8-cdfa-43ef-8c01-142380be7d47)

# <h2>14.1 Most Repeated Adjectives in Positive Reviews (Graph)</h2>
The following bar chart allows us to identify the most common adjectives in positive reviews and provides relevant information on how people express their positive opinions about the movies.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/19b0b4c4-59e8-47f9-98d9-d7259bd1c5e3)

# 15. Most Repeated Adjectives in Negative Reviews
Now, we perform a similar analysis as done for positive reviews, but focusing on negative reviews. We identify the top 10 most repeated adjectives in negative sentiment reviews.
Using the negative_adj_freq object, we assume that it has already been calculated previously to obtain the top 10 most frequent adjectives in negative reviews using most_common(10).
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/a2812439-99a4-4a51-9f64-8c153849f885)

#<h2>15.1 Most Repeated Adjectives in Negative Reviews (Graph)</h2>
The following bar chart allows us to identify the most common adjectives in negative reviews, providing relevant information on how people express their negative opinions about the movies.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/7c1981da-4dee-403f-a9a6-ef90ff3e0404)

# 16. Review Length
Finally, through a box plot, we create a comparative analysis of the length of positive and negative reviews based on the number of words they contain, in order to visualize the distribution of review lengths in both sentiment groups.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/54936e73-82ec-41bd-ba3c-424f631077f7)

# <h2>16.1 Length Distribution (Box Plot)</h2>
Based on the presented diagram, we can assume that positive reviews tend to have slightly longer word lengths compared to negative reviews, as the median of positive reviews is slightly higher than the median of negative reviews.
![image](https://github.com/Nestor-DS/Analisis-de-Sentimientos-en-criticas-de-peliculas/assets/78669277/5830e5ee-6ab7-4042-a128-f15d499fb128)

# References
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
