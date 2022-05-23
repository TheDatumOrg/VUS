import pandas as pd

class Window:
    """ The  class for rolling window feature mapping.
    The mapping converts the original timeseries X into a matrix. 
    The matrix consists of rows of sliding windows of original X. 
    """

    def __init__(self,  window = 100):
        self.window = window
    def convert(self, X):
        n = self.window
        X = pd.Series(X)
        L = []
        if n == 0:
            df = X
        else:
            for i in range(n):
                L.append(X.shift(i))
            df = pd.concat(L, axis = 1)
            df = df.iloc[n-1:]
        return df

