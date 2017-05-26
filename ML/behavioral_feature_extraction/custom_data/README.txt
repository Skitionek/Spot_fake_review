This folder contains the results of the feature extractions.

BBR.npy:
This file contains an array of rows of the below information. As such, the burst review ratio is specific to each user.
user_id
burst_review_ratio

rating_deviation.npy:
This file contains an array of rows of the below information. The average rating deviation is specific to each user.
user_id
average_rating_deviation

user_prod_inburst:
This file does not contain a feature. It contains information on whether a review is included in a burst or not, where each review can be identified by the user id and product id. It is used by the feature extraction algorithms.
user_id
product_id
is_in_a_burst