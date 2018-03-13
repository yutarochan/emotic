import numpy as np 

def divideZero(value_a, value_b): # function to resolve divide by zero error
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide( value_a, value_b )
        result[ ~ np.isfinite( result )] = 0 # accept the value 0 when divided by 0
    return(result)

def zeroloss(self): # return new array with removed element having all zero in y_true
    condition = []
    index = []
    for i in range(y_true.shape[0]):
        new_true = new_pred = list()
        condition = np.logical_and(y_true[i],y_true[i]).sum()
        if (condition==0):
            index = np.asarray(np.append(index,i), dtype = np.int64)
        new_true = np.delete(y_true,index, axis = 0)
        new_pred = np.delete(y_pred,index, axis = 0)
    return(new_true, new_pred)

class mlmetrics:
    # class variable
    real_pos = []       # real positive
    real_neg = []       # real negative
    pred_pos = []       # predicted positive
    pred_neg = []       # predicted negative
    true_pos = []       # true positive
    true_neg = []       # true negative

    def __init__(self, y_true, y_pred):
        # check the matrics size of predicted and true value
        if y_true.shape != y_pred.shape :
            raise ValueError('Size of matrics does not match.')
        # compute all the class variable values
        for i in range(y_true.shape[0]):
            # real values - RP and RN
            self.real_pos = np.asarray(np.append(self.real_pos,np.logical_and(y_true[i], y_true[i]).sum()), dtype=np.int64).reshape(-1,1)
            self.real_neg = np.asarray(np.append(self.real_neg,np.logical_and(np.logical_not(y_true[i]),np.logical_not(y_true[i])).sum()), dtype=np.int64).reshape(-1,1)

            # predicted values - PP and PN
            self.pred_pos = np.asarray(np.append(self.pred_pos,np.logical_and(y_pred[i], y_pred[i]).sum()),dtype=np.int64).reshape(-1,1)
            self.pred_neg = np.asarray(np.append(self.pred_neg,np.logical_and(np.logical_not(y_pred[i]), np.logical_not(y_pred[i])).sum()),dtype=np.int64).reshape(-1,1)

            # true labels - TP and TN
            self.true_pos = np.asarray(np.append(self.true_pos,np.logical_and(y_true[i], y_pred[i]).sum()),dtype=np.int64).reshape(-1,1)
            self.true_neg = np.asarray(np.append(self.true_neg,np.logical_and(np.logical_not(y_true[i]), np.logical_not(y_pred[i])).sum()),dtype=np.int64).reshape(-1,1)


    def accuracy(self): # return the accuracy - example based
        return(np.mean((self.true_pos + self.true_neg)/(self.pred_pos + self.pred_neg)))

    def precision(self): # return precision - example based
        return(np.mean(divideZero(self.true_pos, self.pred_pos)))

    def recall(self): # return precision - example based
        return(np.mean(divideZero(self.true_pos, self.real_pos)))

    def fscore(self,beta = 1): # return f(beta)score - example based : default beta value is 1
        prec, rec, beta_val = precision(), recall(), beta*beta
        return(((1+beta_val)*(prec*rec))/(beta_val*(prec+rec)))

    def microprecision(self): # return micro-precision
        return(self.true_pos.sum()/self.pred_pos.sum())

    def microrecall(self): # return micro-recall
        return(self.true_pos.sum()/self.real_pos.sum())

    def microfscore(self,beta = 1): # return micro-fscore
        prec, rec, beta_val = microprecision(), microrecall(), beta*beta
        return(((1+beta_val)*(prec*rec))/(beta_val*(prec+rec)))

    def macroprecision(self): # return macro-precision
        return(divideZero(self.true_pos, self.pred_pos))

    def macrorecall(self): # return macro-recall
        return(divideZero(self.true_pos, self.real_pos))

    def macrofscore(self,beta = 1): # return macro-fscore
        prec, rec, beta_val = macroprecision(), macrorecall(), beta*beta
        return(np.mean(divideZero(((1+beta_val)*(prec*rec)),(beta_val*(prec+rec)))))

    def hamloss(self): # return hamming loss - example based
        hamloss = []
        for i in range(y_true.shape[0]):
            hamloss = np.asarray(np.append(hamloss,np.logical_xor(y_true[i], y_pred[i]).sum()), dtype=np.int64).reshape(-1,1)
        return((hamloss.sum())/((y_true.shape[0])*(y_true.shape[1])))

    def subset(self): # return subset accuracy - example based
        subset_matrix = []
        for i in range(y_true.shape[0]):
            subset_matrix = np.asarray(np.append(subset_matrix, np.array_equal(y_true[i],y_pred[i])), dtype=np.int64).reshape(-1,1)
        return((subset_matrix.sum())/((y_true.shape[0])*(y_true.shape[1])))
