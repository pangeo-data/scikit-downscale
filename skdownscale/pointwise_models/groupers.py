import numpy as np 
import pandas as pd


class XsdGroupGeneratorBase:
    pass


class PaddedDOYGrouper(XsdGroupGeneratorBase):
    def __init__(self, df, offset=15):
        self.df = df
        self.offset = offset 
        self.max = 365
        self.days_of_year = np.arange(1, 366)
        self.days_of_year_wrapped = np.pad(self.days_of_year, 15, mode='wrap')
        self.n = 1
        
    def __iter__(self):
        self.n = 1
        return self
    
    def __next__(self):
        # n as day of year 
        if self.n > self.max:
            raise StopIteration
            
        i = self.n - 1
        total_days = (2 * self.offset) + 1
        
        first_half = self.days_of_year_wrapped[i:i+self.offset]
        sec_half = self.days_of_year_wrapped[self.n+self.offset:i+total_days]
        all_days = np.concatenate((first_half, np.array([self.n]), sec_half), axis=0)

        assert len(set(all_days)) == total_days, all_days

        result = self.df[self.df.index.dayofyear.isin(all_days)]
        
        self.n += 1
        
        return self.n - 1, result
    
    def mean(self):
        # result = pd.Series()
        list_result = []
        for key, group in self:
            list_result.append(group.mean().values[0])
        result = pd.Series(list_result, index=self.days_of_year)
        return result
