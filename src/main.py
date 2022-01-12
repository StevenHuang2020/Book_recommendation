# -*- encoding: utf-8 -*-
# Date: 02/Jan/2022
# Author: Steven Huang, Auckland, NZ
# License: MIT License
"""
Description: Books recommendation system,
1. Content basesd, recommend_content.py
2. Collaborative Filtering, recommend_collaborative.py

Reference: https://www.asapdevelopers.com/python-recommendation-systems/

Pairwise disntance:
https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity

"""

from recommend_content import rec_content_based
from recommend_collaborative import rec_collaborative


def main():
    rec_content_based()
    # rec_collaborative()


if __name__ == "__main__":
    main()
