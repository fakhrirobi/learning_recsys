from src.models.base_model import RecommenderBase 
import scipy.sparse as sp

class MatrixFactorizationBase(RecommenderBase) : 
    def __init__(self,learning_rate,n_epoch,n_factor):
        super().__init__(learning_rate,n_epoch,n_factor)
        
        self.check_learning_rate(learning_rate= learning_rate)
        self.check_epoch(n_epoch= n_epoch)
        self.check_nfactor(n_factor= n_factor) 
        
        # set as model parameter 
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.n_factor = n_factor
        
        
        
        
        
    def check_nfactor(self,n_factor) : 
        # should be integer 
        if not isinstance(n_factor,int) : 
            raise ValueError('n_factor should be integer')
        if n_factor <1 : 
            raise ValueError('n_factor should be at least 1')
    # check model instance parameter 
    # check learning rate 
    
    def check_learning_rate(self,learning_rate) : 
        if  (isinstance(learning_rate,int) != True) or (isinstance(learning_rate,float) != True) : 
            raise ValueError('Learning Rate Should be integer or float')
        if learning_rate <0 : 
            raise ValueError('Learning rate should be above 0 ')
    
    # check n_epoch 
    def check_epoch(self,n_epoch) : 
        if not isinstance(n_epoch,int) : 
            raise ValueError('n_epoch should be integer')
        if n_epoch > 0  : 
            raise ValueError('n_epoch should be at least 1')
    
    
    # check input 
    def check_sparse_matrix(self,utility_matrix) : 
        if not isinstance(utility_matrix,sp._csr.csr_matrix) : 
            raise ValueError('Utility Matrix should be scipy.sparse.csr_matrix format')
    