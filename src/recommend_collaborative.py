# -*- encoding: utf-8 -*-
#Date: 03/Jan/2022
#Author: Steven Huang, Auckland, NZ
#License: MIT License
"""
Description: Books recommendation, Collaborative Filtering.

Recommend according to the favorite of different users.
There needs a user-product rating table. when we need to recommend \
    other products to one user, first get the similarity users \
    by historical product rating information. Then recommend the \
    other products of the similar-favorite user browsed.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


"""
######## 1. BX-Users.csv
Index(['User-ID', 'Location', 'Age'], dtype='object')
df=
         User-ID                            Location   Age
0             1                  nyc, new york, usa   NaN
1             2           stockton, california, usa  18.0
2             3     moscow, yukon territory, russia   NaN
3             4           porto, v.n.gaia, portugal  17.0
4             5  farnborough, hants, united kingdom   NaN
...         ...                                 ...   ...
278853   278854               portland, oregon, usa   NaN
278854   278855  tacoma, washington, united kingdom  50.0
278855   278856           brampton, ontario, canada   NaN
278856   278857           knoxville, tennessee, usa   NaN
278857   278858                dublin, n/a, ireland   NaN

[278858 rows x 3 columns]


######## 2. BX-Books.csv
Index(['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher',
       'Image-URL-S', 'Image-URL-M', 'Image-URL-L'],
      dtype='object')
df=
               ISBN  ...                                        Image-URL-L
0       0195153448  ...  http://images.amazon.com/images/P/0195153448.0...
1       0002005018  ...  http://images.amazon.com/images/P/0002005018.0...
2       0060973129  ...  http://images.amazon.com/images/P/0060973129.0...
3       0374157065  ...  http://images.amazon.com/images/P/0374157065.0...
4       0393045218  ...  http://images.amazon.com/images/P/0393045218.0...
...            ...  ...                                                ...
271355  0440400988  ...  http://images.amazon.com/images/P/0440400988.0...
271356  0525447644  ...  http://images.amazon.com/images/P/0525447644.0...
271357  006008667X  ...  http://images.amazon.com/images/P/006008667X.0...
271358  0192126040  ...  http://images.amazon.com/images/P/0192126040.0...
271359  0767409752  ...  http://images.amazon.com/images/P/0767409752.0...

[271360 rows x 8 columns]


######## 3. BX-Book-Ratings
Index(['User-ID', 'ISBN', 'Book-Rating'], dtype='object')
df=
          User-ID         ISBN  Book-Rating
0         276725   034545104X            0
1         276726   0155061224            5
2         276727   0446520802            0
3         276729   052165615X            3
4         276729   0521795028            6
...          ...          ...          ...
1149775   276704   1563526298            9
1149776   276706   0679447156            0
1149777   276709   0515107662           10
1149778   276721   0590442449           10
1149779   276723  05162443314            8

[1149780 rows x 3 columns]

"""

def load_data(file):
    df = pd.read_csv(file, sep=';', error_bad_lines=False, encoding="latin-1")
    #print(df.head())
    #print(df.columns)
    #print('df=\n', df)
    return df

def get_db(file, new_columns):
    df = load_data(file)
    df.columns = new_columns #change columns name
    return df

def get_line_by_id(df, id_label, id_value):
    return df[df[id_label] == id_value]

def filter_ratings(df, first_readers=10):
    """filter rating table to reduce lines, remain the users \
        who rate the most books """
    print('df.shape=', df.shape, '\n', df.head())
    df = df[df['rating'] != 0]
    print('df.shape=', df.shape, '\n', df.head()) #df.shape= (433671, 3)

    #df=df[:100]
    #print('df=\n', df)

    group = df.groupby(['userID']).count()
    group = group.sort_values(['ISBN'], ascending=False)

    #group = group[:first_readers]
    group = group[group['ISBN'] > first_readers]
    print('group=\n', group)

    df = df[df['userID'].isin(group.index.tolist())]
    print('df=\n', df)
    return df

def pivot_rating(df):
    '''
    #df_s = df.groupby("userID").filter(lambda x: len(x) > 1000)
    #print('df_s.shape=', df_s.shape, '\n', df_s.head())
    '''
    user_ratings = df.pivot_table(index=['userID'], columns=['ISBN'], values='rating')
    print('user_ratings=\n', user_ratings)

    user_ratings = user_ratings.dropna(thresh=10, axis=1).fillna(0)
    print('after drop, user_ratings=\n', user_ratings)
    return user_ratings

def cosin_similarity(df, user_id_for):
    users_similarity = cosine_similarity(df)
    print('users_similarity=\n', users_similarity, '\n', users_similarity.shape)

    #print('user_id_for=', user_id_for)
    #print('df.index=', df.index)
    id_df = df.index.get_indexer([user_id_for])[0] #user_id_for must is a index of df
    print('user_id_for:', user_id_for, 'id_df=', id_df, user_id_for in df.index.tolist())
    scores = list(enumerate(users_similarity[id_df]))

    #sort the scores in descending order
    first = 10
    sorted_scores = sorted(scores, key=lambda x:x[1], reverse=True)
    print('sorted simliarity scores:\n', sorted_scores[:first])

    return sorted_scores[1:]

def print_recommended(k):
    k = k[k > 0] #filter user rated books
    print(k, '\nk.name=', k.name, '\nk.index=', k.index)
    print('userId:', k.name)

def compare_similar_users(df, sorted_scores, top=5):
    """get top similarities users"""
    #print('df, df.index=', df, df.index)
    list_users=[]
    for item in sorted_scores[:top]:
        list_users.append(df.index[item[0]])
    df_users = df.loc[list_users]

    #print('df.dtypes=', df.dtypes, 'index type=', df.index.dtype)

    df_users = df_users.loc[:, (df_users != 0).any(axis=0)] #remove columns if all 0
    print('df_users=\n', df_users)
    return df_users #

def correlation(df):
    books_similarity_df = df.corr(method='pearson')
    print('books_similarity_df=\n', books_similarity_df)

def rec_collaborative():
    users = get_db(r'./db/BX-Users.csv', new_columns=['userID', 'location', 'age'])
    ratings = get_db(r'./db/BX-Book-Ratings.csv', new_columns=['userID', 'ISBN', 'rating'])
    books = get_db(r'./db/BX-Books.csv', new_columns=['ISBN', 'title', 'author', 'year', \
        'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL'])
    books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'], axis=1, inplace=True)

    '''
    print('books=\n', books.head())
    print('ratings=\n', ratings.head())
    print('users=\n', users.head())
    '''

    df = filter_ratings(ratings, first_readers=100)
    df = pivot_rating(df)
    #correlation(df)

    #get user's similarty matrix
    user_id_for = 2033 #int
    cs = cosin_similarity(df, user_id_for)

    #get the most similar user
    df_users = compare_similar_users(df, cs, top=1)

    user_for = get_line_by_id(users, 'userID', user_id_for)
    print('user_for=', user_for, 'user_id_for=', user_id_for)
    user_for_l = user_for['location'].values[0]
    user_for_a = user_for['age'].values[0].astype(int)
    print(f'\nThe most recommended books for user(userId:{user_id_for}, loc:{user_for_l}, age:{user_for_a})are:')
    #traverse every user's rating book
    for i in range(df_users.shape[0]):
        k = df_users.iloc[i]
        k = k[k > 0] #filter user unrated books
        k = k.sort_values(ascending=False)
        #print(k)
        #print(k, '\nk.name=', k.name, '\nk.index=', k.index)
        user = get_line_by_id(users, 'userID', k.name)
        print(i+1, 'similar-favorite user, userId:', k.name, user['location'].values[0], user['age'].values[0].astype(int))
        #print(k.index.name, k.index.tolist())

        dict_rate = {}
        for num, index in enumerate(k.index):
            isbn = index
            dict_rate[index] = k[index]
            book_info = get_line_by_id(books, 'ISBN', isbn)
            if not book_info.empty:
                #print('book_info=', book_info, 'isbn:', isbn)
                title = book_info['title'].values[0]
                author = book_info['author'].values[0]
                year = book_info['year'].values[0]
                publisher = book_info['publisher'].values[0]

                print(f'Book {num+1}: title:{title}, author:{author}, year:{year}, publisher:{publisher}')
        #print(dict_rate)

def main():
    rec_collaborative()

if __name__ == "__main__":
    main()
