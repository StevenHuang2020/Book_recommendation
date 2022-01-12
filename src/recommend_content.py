# -*- encoding: utf-8 -*-
# Date: 03/Jan/2022
# Author: Steven Huang, Auckland, NZ
# License: MIT License
"""
Description: Books recommendation, content basesd.

Recommend only according to the similarity of different books.
Get the similarity matrix of one book and all other rows, then
recommend according to the similarity matrix.

"""

from numpy.core.numeric import NaN
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def load_data(file):
    df = pd.read_csv(file, )
    # print(df.head())
    print(df.columns)
    print('df=\n', df)
    return df


def combine_features(data, columns):  # columns=['title', 'authors']
    """combine the selected important columns in to a single string.
    'title' 'authors' ..., these columns/attributes are the
    similarity related key features. You can also use
    one column as the key feature.
    """

    features = []
    for i in range(data.shape[0]):
        str_tmp = ''

        # data['title'][i] + ' ' + data['authors'][i]
        for col in columns:
            if data[col][i] is not NaN:
                str_tmp += (data[col][i] + ' ')

        features.append(str_tmp)
    return features


def get_one_book(data, num=0):
    return data.iloc[num]


def calculate_cs(df_column):
    """get cosine similarity matrix, df_column: one column of
    a dataframe, for example: df['publisher'] """

    vectorizer = CountVectorizer()

    # covert the text from the new column "combine_features"
    count_matrix = vectorizer.fit_transform(df_column)
    # print(vectorizer.get_feature_names())
    # print('count_matrix=\n', count_matrix, type(count_matrix))
    # print("count_matrix.shape=", count_matrix.shape)
    print('count_matrix=\n', count_matrix.toarray())

    # get the cosine similarity matrix fromt the count matrix
    cs = cosine_similarity(count_matrix)
    print('cosine similarity=\n', cs, type(cs))
    return cs


def recommend_for_user(cs, df, user_id=0, top=10):
    """
    recommend for one user

    Args:
        cs: books similarity matrix
        df: data
        user_id (int): user id, must belongto the index of df
        top (int): recommed top final results
    """

    one = get_one_book(df, user_id)
    book_title = one['title']
    book_id = one['id']  # 'book_id'

    print('book_title=', book_title, 'book_id=', book_id)

    # create a list of tuples int the form (book_id, similarity_score)
    # print('book simliarity:', cs[book_id])
    scores = list(enumerate(cs[book_id]))

    # show the first 100 values
    print('book simliarity scores:\n', scores[:100])

    # sort the scores in descending order
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print('sorted book simliarity scores:\n', sorted_scores[:10])

    # create a loop to show the first 10 books from the sorted list
    print(f'\nThe {top} most recommended books to {book_title} are:')

    for i, item in enumerate(sorted_scores[1:top + 1], start=1):
        # print (i, item)
        book = df[df['id'] == item[0]]
        # book['isbn13'] = df['isbn13'].fillna(0)
        # book['isbn13'] = df['isbn13'].astype(int)

        book_id = book['id'].values[0]
        title = book['title'].values[0]
        year = book['original_publication_year'].values[0]
        authors = book['authors'].values[0]
        average_rating = book['average_rating'].values[0]
        # isbn13 = book['isbn13'].values[0]
        # language_code = book['language_code'].values[0]
        print(i, book_id, title, year, authors, average_rating)


def do_dataset1():
    file = r'./db/books.csv'
    df = load_data(file)

    # combine key features for similarity calculation
    columns = ['title', 'authors']
    df['combine_features'] = combine_features(df, columns=columns)

    cs = calculate_cs(df['combine_features'])
    recommend_for_user(cs, df, 2)


def do_dataset2():
    """This is another case"""
    file = r'./db/books_new.csv'
    df = load_data(file)

    columns = ['Author', 'Genre']
    df['combine_features'] = combine_features(df, columns=columns)

    cs = calculate_cs(df['combine_features'])

    # Select a book and recommend similar books with it
    book_id = 208
    one = get_one_book(df, book_id)
    book_title = one['Title']

    print('book_title=', book_title, 'book_id=', book_id)

    # create a list of tuples int the form (book_id, similarity_score)
    # print('book simliarity:', cs[book_id])
    scores = list(enumerate(cs[book_id]))

    # show the first 100 values
    print('book simliarity scores:\n', scores[:100])

    # sort the scores in descending order
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print('sorted book simliarity scores:\n', sorted_scores[:10])

    # create a loop to show the first 10 books from the sorted list
    first = 10
    print(f'\nThe {first} most recommended books to <{book_title}> are:')

    for i, item in enumerate(sorted_scores[1:first + 1], start=1):
        # print (i, item)
        book_id = item[0]
        book = df.iloc[book_id]

        title = book['Title']
        author = book['Author']
        genre = book['Genre']
        publisher = book['Publisher']

        print(i, 'Id:', book_id, 'Title:', title, 'Author:', author, 'Genre:',
              genre, 'Publisher:', publisher)


def rec_content_based():
    do_dataset1()
    # do_dataset2()
