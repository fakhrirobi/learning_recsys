import scipy.sparse as sp 
import numpy as np 

from src.models.

# SVDClass 
class FunkSVD : 
    
    def __init__(self,n_factors=50, n_epoch=20, lr=0.001, lamba_reg=0.002 ) -> None:
        """Initializes Model Hyperparameter / Configuration

        Parameters
        ----------
        n_factors : int
            Number of latent factor to use.
        n_epoch : int
            Number of iterations during model training (epochs)
        lr : float
            learning rate 
        lamba_reg : float
            regularization strength for objective function / model parameter 

        Returns
        -------
        - 
        """
        self.n_factors = n_factors 
        self.n_epoch = n_epoch 
        self.lr = lr 
        self.lamba_reg = lamba_reg 
         
    
    def generate_mapping(self) :
        """Function to generate mapping on userId and itemId"""
        self.user_to_id = { user_id : idx for idx,user_id in 
                                   enumerate(self.utility_matrix[self.user_column])}
         
        self.id_to_user = { idx : user_id for idx,user_id in 
                                   enumerate(self.utility_matrix[self.user_column])}
        
        self.item_to_id = { item_id : idx for idx,item_id in 
                                   enumerate(self.utility_matrix[self.item_column])}
         
        self.id_to_item = { idx : item_id for idx,item_id in 
                                   enumerate(self.utility_matrix[self.item_column])}
        

        
        
    def initialize_parameters(self) : 
        """Initializes biases and latent factor matrices.

        Parameters
        ----------
        n_users : int
            Number of unique users.
        n_items : int
            Number of unique items.
        n_factors : int
            Number of factors.

        Returns
        -------
        bu : numpy.array
            User biases vector.
        bi : numpy.array
            Item biases vector.
        pu : numpy.array
            User latent factors matrix.
        qi : numpy.array
            Item latent factors matrix.
        """
        bu = np.zeros(self.n_users)
        bi = np.zeros(self.n_items)

        pu = np.random.normal(0, .1, (self.n_users,self.n_factors))
        qi = np.random.normal(0, .1, (self.n_items,self.n_factors))


        return bu,bi,pu,qi
    

            
    # def update_parameter(self,n_epoch,global_mean,bu,bi,pu,qi) : 
    #     """
    #     Function to update parameter with gradient descent
    #     Parameters
    #     ----------
    #     n_epochs : int 
    #         Number of epochs
    #     global_mean : float 
    #         global mean for computing baseline prediction
    #     bu : numpy.array
    #         User biases vector.
    #     bi : numpy.array
    #         Item biases vector.
    #     pu : numpy.array
    #         User latent factors matrix.
    #     qi : numpy.array
    #         Item latent factors matrix.
        

    #     Returns
    #     -------
    #     bu : numpy.array
    #         Updated User biases vector.
    #     bi : numpy.array
    #         Updated Item biases vector.
    #     pu : numpy.array
    #         Updated User latent factors matrix.
    #     qi : numpy.array
    #         Updated Item latent factors matrix.
    #     loss_per_epoch : list 
    #         Average loss per epoch
    #     """
    #     # copy parameter first to avoid overwriting 
    #     bu = bu.copy()
    #     bi = bi.copy()
    #     pu = pu.copy()
    #     qi = qi.copy()

    #     # run update parameter 
    #     loss_per_epoch = []
    #     for _ in range(n_epoch) :
    #         # add empty list to add avg loss per epoch 
    #         avg_loss_training_samples = []
    #         # iterate all over training data 
    #         for training_samples in self.utility_matrix : 
                


    #     # return bu,bi,pu,qi,loss_per_epoch
    
    def fit(self,utility_matrix) : 
        """
        

        Parameters : 
        -----------
            utility_matrix (scipy.sparse._csr.csr_matrix): _description_

            
        Returns : 
        -----------
        self (object)
        """
        # check utility_matrix input 
        self.check_sparse_matrix(utility_matrix)
        
        self.utility_matrix = utility_matrix 
        
        # initialize parameters 
        bu,bi,pu,qi = self.initialize_parameters()
        
        # extract global mean 
        self.global_mean = self.utility_matrix.sum()/self.utility_matrix.nnz
        
        # update parameters
        bu_updated, bi_updated, pu_updated,qi_updated,loss = self.update_parameters(n_epoch = self.n_epochs, 
                                                                                    global_mean = self.global_mean,
                                                                                    bu= bu, bi= bi, pu= pu, qi= qi)
        
        
        
        
        

    

