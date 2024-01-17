"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb;

class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        print("reset parameters for ScaledEmbedding")
        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=32,
        layer_sizes=[96, 64],
        sparse=False,
        embedding_sharing=True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************

        # init weight for MultiTaskNet
        self.embedding_sharing = embedding_sharing

        # Initialize embeddings
        # Initialize embeddings for factorization task
        self.user_embeddings_factorization = ScaledEmbedding(
            num_users, embedding_dim, sparse=sparse
        )
        self.item_embeddings_factorization = ScaledEmbedding(
            num_items, embedding_dim, sparse=sparse
        )

        # Initialize embeddings for regression task, separate from factorization if not sharing
        if not embedding_sharing:
            self.user_embeddings_regression = ScaledEmbedding(
                num_users, embedding_dim, sparse=sparse
            )
            self.item_embeddings_regression = ScaledEmbedding(
                num_items, embedding_dim, sparse=sparse
            )
        else:
            # If sharing embeddings, point to the same embeddings used in factorization
            self.user_embeddings_regression = self.user_embeddings_factorization
            self.item_embeddings_regression = self.item_embeddings_factorization
            # Reset parameters of regression embeddings


        self.user_biases = ZeroEmbedding(num_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse)

        self.regression = nn.Sequential(
            nn.Linear(3 * embedding_dim, layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(layer_sizes[1], 1),
        )

        # ********************************************************
        # ********************************************************
        # ********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        # ********************************************************
        # ******************* YOUR CODE HERE *********************
        # ********************************************************
        # Get the embeddings for factorization task
        user_embedding_factorization = self.user_embeddings_factorization(user_ids)
        item_embedding_factorization = self.item_embeddings_factorization(item_ids)

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        # Compute the dot product (interaction term) plus bias terms
        interaction = (user_embedding_factorization * item_embedding_factorization).sum(
            dim=1
        )
        prediction = interaction + user_bias + item_bias

        # Apply the sigmoid function to get the probability p_ij
        predictions = torch.sigmoid(prediction)

        # Get the embeddings for regression task
        user_embedding_regression = self.user_embeddings_regression(user_ids)
        item_embedding_regression = self.item_embeddings_regression(item_ids)

        # Concatenate the user and item embeddings with their element-wise product
        combined_features = torch.cat(
            (
                user_embedding_regression,
                item_embedding_regression,
                user_embedding_regression * item_embedding_regression,
            ),
            dim=1,
        )
        # Pass through regression layers
        score = self.regression(combined_features).squeeze()
        # Pass through the final output layer to get the score
        # pdb.set_trace()

        # ********************************************************
        # ********************************************************
        # ********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        # print(f"shape of predictions: {predictions.shape} and score: {score.shape}")
        return predictions, score
