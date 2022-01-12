## Book recommendation

The key to the recommendation system is the similarity calculation. \
The [cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html), and other pairwise metrics algorithms can refer to [here](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise). \
The similarity matrix is a symmetric matrix, one line of similarity matrix means \
its similarity or difference with all other lines. One example looks like below:

```
cosine similarity=
 [[1.         0.13801311 0.         ... 0.23904572 0.25197632 0.21821789]
 [0.13801311 1.         0.         ... 0.11547005 0.12171612 0.10540926]
 [0.         0.         1.         ... 0.         0.         0.        ]
 ...
 [0.23904572 0.11547005 0.         ... 1.         0.21081851 0.18257419]
 [0.25197632 0.12171612 0.         ... 0.21081851 1.         0.19245009]
 [0.21821789 0.10540926 0.         ... 0.18257419 0.19245009 1.        ]]
 ```
 
#### 1. The output of content-based recommendation
```
The 10 most recommended books to The Hunger Games (The Hunger Games, #1) are:
1 24 Harry Potter and the Goblet of Fire (Harry Potter, #4) 2000.0 J.K. Rowling, Mary GrandPré 4.53
2 2100 Batman: The Dark Knight Returns #1 1986.0 Frank Miller 4.21
3 22 The Lovely Bones 2002.0 Alice Sebold 3.77
4 23 Harry Potter and the Chamber of Secrets (Harry Potter, #2) 1998.0 J.K. Rowling, Mary GrandPré 4.37
5 26 The Da Vinci Code (Robert Langdon, #2) 2003.0 Dan Brown 3.79
6 20 Mockingjay (The Hunger Games, #3) 2010.0 Suzanne Collins 4.03
7 17 Catching Fire (The Hunger Games, #2) 2009.0 Suzanne Collins 4.3
8 3274 Hop On Pop 1963.0 Dr. Seuss 3.95
9 421 The Paris Wife 2011.0 Paula McLain 3.79
10 3752 Poirot Investiga (Hércules Poirot, #3) 1924.0 Agatha Christie 4.07
```

#### 2. The output of collaborative filtering
```
The most recommended books for user(userId:2033, loc:omaha, nebraska, usa, age:27)are:
1 similar-favorite user, userId: 179978 perry, new york, usa 28
Book 1: title:Christmas Box (Christmas Box Trilogy), author:Richard Paul Evans, year:1995, publisher:Simon &amp; Schuster
Book 2: title:Harry Potter and the Sorcerer's Stone (Book 1), author:J. K. Rowling, year:1998, publisher:Scholastic
Book 3: title:Harry Potter and the Order of the Phoenix (Book 5), author:J. K. Rowling, year:2003, publisher:Scholastic
Book 4: title:Harry Potter and the Goblet of Fire (Book 4), author:J. K. Rowling, year:2000, publisher:Scholastic
Book 5: title:Harry Potter and the Prisoner of Azkaban (Book 3), author:J. K. Rowling, year:1999, publisher:Scholastic
Book 6: title:Harry Potter and the Chamber of Secrets (Book 2), author:J. K. Rowling, year:1999, publisher:Scholastic
Book 7: title:The Cat in the Hat, author:Dr. Seuss, year:1957, publisher:Random House Books for Young Readers
Book 8: title:The Tao of Pooh, author:Benjamin Hoff, year:1983, publisher:Penguin Books
Book 9: title:The Giving Tree, author:Shel Silverstein, year:1964, publisher:HarperCollins Publishers
Book 10: title:Love You Forever, author:Robert N. Munsch, year:1986, publisher:Firefly Books Ltd
Book 11: title:Melody (Logan), author:V.C. Andrews, year:1996, publisher:Pocket
```