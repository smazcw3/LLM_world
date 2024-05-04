from math import ceil
from tqdm import trange
import numpy as np
from itertools import islice
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, dok_matrix

class BPR:
    """
    Bayesian Personalized Ranking (BPR) for implicit feedback data.

    Attributes:
        learning_rate (float): Learning rate for gradient descent. Default is 0.01.
        n_factors (int): Number/dimension of user and item latent factors. Default is 15.
        n_iters (int): Number of iterations to train the algorithm. Default is 10.
        batch_size (int): Batch size for batch gradient descent. Default is 1000.
        reg (float): Regularization term for the user and item latent factors. Default is 0.01.
        seed (int): Seed for the randomly initialized user, item latent factors. Default is 1234.
        verbose (bool): Whether to print progress bar while training. Default is True.
        user_factors (2d ndarray): User latent factors learnt. Shape is [n_users, n_factors].
        item_factors (2d ndarray): Item latent factors learnt. Shape is [n_items, n_factors].

    Methods:
        fit(ratings): Train the model with the given ratings.
        predict(): Obtain the predicted ratings for every users and items.
        recommend(ratings, N): Returns the top N ranked items for given user id.
        get_similar_items(N, item_ids): Return the top N similar items for itemid.

    References:
        S. Rendle, C. Freudenthaler, Z. Gantner, L. Schmidt-Thieme
        Bayesian Personalized Ranking from Implicit Feedback
        - https://arxiv.org/abs/1205.2618
    """
    def __init__(self, learning_rate = 0.01, n_factors = 15, n_iters = 10, 
                 batch_size = 1000, reg = 0.01, seed = 1234, verbose = True):
        self.reg = reg
        self.seed = seed
        self.verbose = verbose
        self.n_iters = n_iters
        self.n_factors = n_factors
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # to avoid re-computation at predict
        self._prediction = None
        
    def fit(self, ratings):
        """
        Trains the model with the given ratings.

        This method initializes random weights for user and item factors. It then performs
        batch gradient descent over a number of iterations to optimize the factors based on
        the input ratings.

        Parameters:
            ratings (scipy.sparse.csr_matrix): A sparse matrix of user-item interactions. Shape [n_users, n_items]

        Returns:
            self: Returns the instance of the model.
        """
        indptr = ratings.indptr
        indices = ratings.indices
        n_users, n_items = ratings.shape
        
        # ensure batch size makes sense, since the algorithm involves
        # for each step randomly sample a user, thus the batch size
        # should be smaller than the total number of users or else
        # we would be sampling the user with replacement
        batch_size = self.batch_size
        if n_users < batch_size:
            batch_size = n_users
            sys.stderr.write('WARNING: Batch size is greater than number of users,'
                             'switching to a batch size of {}\n'.format(n_users))

        batch_iters = n_users // batch_size
        
        # initialize random weights
        rstate = np.random.RandomState(self.seed)
        self.user_factors = rstate.normal(size = (n_users, self.n_factors))
        self.item_factors = rstate.normal(size = (n_items, self.n_factors))
        
        # progress bar for training iteration if verbose is turned on
        loop = range(self.n_iters)
        if self.verbose:
            loop = trange(self.n_iters, desc = self.__class__.__name__)
        
        for _ in loop:
            for _ in range(batch_iters):
                sampled = self._sample(n_users, n_items, indices, indptr)
                sampled_users, sampled_pos_items, sampled_neg_items = sampled
                self._update(sampled_users, sampled_pos_items, sampled_neg_items)

        return self
    
    def _sample(self, n_users, n_items, indices, indptr):
        """
        Samples batches of random triplets u, i, j.

        This method is used during the training phase to generate random triplets of users, positive items, and negative items.
        For each user, it randomly selects a positive item (an item the user has interacted with) and a negative item (an item the user has not interacted with).

        Parameters:
            n_users (int): Total number of users.
            n_items (int): Total number of items.
            indices (1d ndarray): Indices of the ratings (interactions) for each user.
            indptr (1d ndarray): Index pointers for the start of each user in the indices array.

        Returns:
            tuple: A tuple containing three 1D numpy arrays, each of size 'batch_size'. The first array contains the user indices, the second contains the positive item indices, and the third contains the negative item indices.
        """
        sampled_pos_items = np.zeros(self.batch_size, dtype = np.int)
        sampled_neg_items = np.zeros(self.batch_size, dtype = np.int)
        sampled_users = np.random.choice(
            n_users, size = self.batch_size, replace = False)

        for idx, user in enumerate(sampled_users):
            pos_items = indices[indptr[user]:indptr[user + 1]]
            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(n_items)
            while neg_item in pos_items:
                neg_item = np.random.choice(n_items)

            sampled_pos_items[idx] = pos_item
            sampled_neg_items[idx] = neg_item

        return sampled_users, sampled_pos_items, sampled_neg_items
                
    def _update(self, u, i, j):
        """
        Performs the gradient descent update for a single triplet.

        This method updates the user and item factors using gradient descent for the given triplet.

        Parameters:
            u (1d ndarray): Array of user indices.
            i (1d ndarray): Array of positive item indices for the users in 'u'.
            j (1d ndarray): Array of negative item indices for the users in 'u'.

        Returns:
            self: Returns the instance of the model.
        """
        user_u = self.user_factors[u]
        item_i = self.item_factors[i]
        item_j = self.item_factors[j]
        
        # decompose the estimator, compute the difference between
        # the score of the positive items and negative items; a
        # naive implementation might look like the following:
        # r_ui = np.diag(user_u.dot(item_i.T))
        # r_uj = np.diag(user_u.dot(item_j.T))
        # r_uij = r_ui - r_uj
        
        # however, we can do better, so
        # for batch dot product, instead of doing the dot product
        # then only extract the diagonal element (which is the value
        # of that current batch), we perform a hadamard product, 
        # i.e. matrix element-wise product then do a sum along the column will
        # be more efficient since it's less operations
        # http://people.revoledu.com/kardi/tutorial/LinearAlgebra/HadamardProduct.html
        # r_ui = np.sum(user_u * item_i, axis = 1)
        #
        # then we can achieve another speedup by doing the difference
        # on the positive and negative item up front instead of computing
        # r_ui and r_uj separately, these two idea will speed up the operations
        # from 1:14 down to 0.36
        r_uij = np.sum(user_u * (item_i - item_j), axis = 1)
        sigmoid = np.exp(-r_uij) / (1.0 + np.exp(-r_uij))
        
        # repeat the 1 dimension sigmoid n_factors times so
        # the dimension will match when doing the update
        sigmoid_tiled = np.tile(sigmoid, (self.n_factors, 1)).T

        # update using gradient descent
        grad_u = sigmoid_tiled * (item_j - item_i) + self.reg * user_u
        grad_i = sigmoid_tiled * -user_u + self.reg * item_i
        grad_j = sigmoid_tiled * user_u + self.reg * item_j
        self.user_factors[u] -= self.learning_rate * grad_u
        self.item_factors[i] -= self.learning_rate * grad_i
        self.item_factors[j] -= self.learning_rate * grad_j
        return self

    def predict(self):
        """
        Obtain the predicted ratings for every user and item by doing a dot product of the learnt user and item vectors.
        The result will be cached to avoid re-computing it every time we call predict, thus there will only be an overhead the first
        time we call it. Note, ideally you probably don't need to compute this as it returns a dense matrix and may take up huge amounts of
        memory for large datasets.

        Returns:
            2d ndarray: Predicted ratings for all users and items. Shape is [n_users, n_items].
        """
        if self._prediction is None:
            self._prediction = self.user_factors.dot(self.item_factors.T)

        return self._prediction

    def _predict_user(self, user):
        """
        Returns the predicted ratings for the specified user.

        This method computes the dot product of the user factors and item factors for the specified user to get the predicted ratings.

        Parameters:
            user (int): Index of the user.

        Returns:
            1d ndarray: Predicted ratings for the user across all items.
        """
        user_pred = self.user_factors[user].dot(self.item_factors.T)
        return user_pred

    def recommend(self, ratings, N = 5):
        """
        Returns the top N ranked items for each user, excluding the items that the user has already interacted with.

        Parameters:
            ratings (scipy.sparse.csr_matrix): A sparse matrix of user-item interactions. Shape [n_users, n_items]
            N (int, optional): The number of top ranked items to return for each user. Default is 5.

        Returns:
            2d ndarray: A 2D array where each row contains the top N ranked item indices for each user. Shape is [n_users, N].
        """
        n_users = ratings.shape[0]
        recommendation = np.zeros((n_users, N), dtype = np.uint32)
        for user in range(n_users):
            top_n = self._recommend_user(ratings, user, N)
            recommendation[user] = top_n

        return recommendation

    def _recommend_user(self, ratings, user, N):
        """
        Returns the top N ranked items for a specific user, excluding the items that the user has already interacted with.

        Parameters:
            ratings (scipy.sparse.csr_matrix): A sparse matrix of user-item interactions. Shape [n_users, n_items]
            user (int): Index of the user for which recommendations need to be generated.
            N (int): The number of top ranked items to return for the user.

        Returns:
            list: A list of top N ranked item indices for the user.
        """
        scores = self._predict_user(user)

        # compute the top N items, removing the items that the user already liked
        # from the result and ensure that we don't get out of bounds error when 
        # we ask for more recommendations than that are available
        liked = set(ratings[user].indices)
        count = N + len(liked)
        if count < scores.shape[0]:

            # when trying to obtain the top-N indices from the score,
            # using argpartition to retrieve the top-N indices in 
            # unsorted order and then sort them will be faster than doing
            # straight up argort on the entire score
            # http://stackoverflow.com/questions/42184499/cannot-understand-numpy-argpartition-output
            ids = np.argpartition(scores, -count)[-count:]
            best_ids = np.argsort(scores[ids])[::-1]
            best = ids[best_ids]
        else:
            best = np.argsort(scores)[::-1]

        top_n = list(islice((rec for rec in best if rec not in liked), N))
        return top_n
    
    def get_similar_items(self, N = 5, item_ids = None):
        """
        Returns the top N similar items for each specified item ID. The similarity is computed using cosine distance.

        If no specific item IDs are provided, this method will compute similar items for all items.

        Parameters:
            N (int, optional): The number of similar items to return for each item. Default is 5.
            item_ids (1d array-like, optional): The IDs of the items to compute similar items for. If None, compute similar items for all items.

        Returns:
            2d ndarray: A 2D array where each row contains the top N similar item indices for each item. Shape is [number of query item_ids, N].
        """
        # cosine distance is proportional to normalized euclidean distance,
        # thus we normalize the item vectors and use euclidean metric so
        # we can use the more efficient kd-tree for nearest neighbor search;
        # also the item will always to nearest to itself, so we add 1 to 
        # get an additional nearest item and remove itself at the end
        normed_factors = normalize(self.item_factors)
        knn = NearestNeighbors(n_neighbors = N + 1, metric = 'euclidean')
        knn.fit(normed_factors)

        # returns a distance, index tuple,
        # we don't actually need the distance
        if item_ids is not None:
            normed_factors = normed_factors[item_ids]

        _, items = knn.kneighbors(normed_factors)
        similar_items = items[:, 1:].astype(np.uint32)
        return similar_items
      

def create_matrix(data, users_col, items_col, ratings_col, threshold = None):
    """
    Creates the sparse user-item interaction matrix. If the data is not in the format where the interaction only
    contains the positive items (indicated by 1), then use the threshold parameter to determine which items are considered positive.

    Parameters:
        data (DataFrame): Implicit rating data.
        users_col (str): User column name.
        items_col (str): Item column name.
        ratings_col (str): Implicit rating column name.
        threshold (int, optional): Threshold to determine whether the user-item pair is a positive feedback. Default is None.

    Returns:
        tuple: A tuple containing the following elements:
            ratings (scipy.sparse.csr_matrix): User/item ratings matrix of shape [n_users, n_items].
            data (DataFrame): Implicit rating data that retains only the positive feedback (if specified to do so).
    """
    if threshold is not None:
        data = data[data[ratings_col] >= threshold]
        data[ratings_col] = 1

    # this ensures each user has at least 2 records to construct a valid
    # train and test split in downstream process, note we might purge
    # some users completely during this process
    data_user_num_items = (data
                         .groupby('user_id')
                         .agg(**{'num_items': ('item_id', 'count')})
                         .reset_index())
    data = data.merge(data_user_num_items, on='user_id', how='inner')
    data = data[data['num_items'] > 1]
    
    for col in (items_col, users_col, ratings_col):
        data[col] = data[col].astype('category')

    ratings = csr_matrix((data[ratings_col],
                          (data[users_col].cat.codes, data[items_col].cat.codes)))
    ratings.eliminate_zeros()
    return ratings, data


def create_train_test(ratings, test_size = 0.2, seed = 1234):
    """
    Splits the user-item interactions matrix into train and test sets by removing some of the interactions from every user and pretending
    that we never seen them.

    Parameters:
        ratings (scipy.sparse.csr_matrix): The user-item interactions matrix of shape [n_users, n_items].
        test_size (float, optional): Proportion of the user-item interactions for each user in the dataset to move to the test set. Default is 0.2.
        seed (int, optional): Seed for reproducible random splitting the data into train/test set. Default is 1234.

    Returns:
        tuple: A tuple containing the following elements:
            train (scipy.sparse.csr_matrix): Training set of shape [n_users, n_items].
            test (scipy.sparse.csr_matrix): Test set of shape [n_users, n_items].
    """
    assert test_size < 1.0 and test_size > 0.0

    # Dictionary Of Keys based sparse matrix is more efficient
    # for constructing sparse matrices incrementally compared with csr_matrix
    train = ratings.copy().todok()
    test = dok_matrix(train.shape)
    
    # for all the users assign randomly chosen interactions
    # to the test and assign those interactions to zero in the training;
    # when computing the interactions to go into the test set, 
    # remember to round up the numbers (e.g. a user has 4 ratings, if the
    # test_size is 0.2, then 0.8 ratings will go to test, thus we need to
    # round up to ensure the test set gets at least 1 rating)
    rstate = np.random.RandomState(seed)
    for u in range(ratings.shape[0]):
        split_index = ratings[u].indices
        n_splits = ceil(test_size * split_index.shape[0])
        test_index = rstate.choice(split_index, size = n_splits, replace = False)
        test[u, test_index] = ratings[u, test_index]
        train[u, test_index] = 0
    
    train, test = train.tocsr(), test.tocsr()
    return train, test


def auc_score(model, ratings):
    """
    Computes the Area Under the ROC Curve (AUC) for the predictions made by the model. The function computes the AUC
    for every user's prediction and actual interaction and takes the average for all users, hence it could also be
    referred to as the mean AUC score.

    Parameters:
        model (BPR): Trained BPR model.
        ratings (scipy.sparse.csr_matrix): Sparse matrix of user-item interactions of shape [n_users, n_items].

    Returns:
        float: Mean AUC score for all users. The value is between 0.0 and 1.0.
    """
    auc = 0.0
    n_users, n_items = ratings.shape
    for user, row in enumerate(ratings):
        y_pred = model._predict_user(user)
        y_true = np.zeros(n_items)
        y_true[row.indices] = 1
        auc += roc_auc_score(y_true, y_pred)

    auc /= n_users
    return auc