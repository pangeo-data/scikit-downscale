import numpy as np
import pandas as pd


class SkdownscaleGroupGeneratorBase:
    pass


def MONTH_GROUPER(x):
    return x.month

def DAY_GROUPER(x):
    return x.day

class PaddedDOYGrouper(SkdownscaleGroupGeneratorBase):
    def __init__(self, df, offset=15):
        self.df = df
        self.offset = offset
        self.max = 365
        self.days_of_year = np.arange(1, 366)
        self.days_of_year_wrapped = np.pad(self.days_of_year, 15, mode="wrap")
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

        # create day groups with +/- days
        # number of days defined by offset
        first_half = self.days_of_year_wrapped[i : i + self.offset]
        sec_half = self.days_of_year_wrapped[self.n + self.offset : i + total_days]
        all_days = np.concatenate((first_half, np.array([self.n]), sec_half), axis=0)

        assert len(set(all_days)) == total_days, all_days
        if len(set(all_days)) != total_days:
            raise ValueError("day groups do not contain the correct set of days")

        result = self.df[self.df.index.dayofyear.isin(all_days)]

        self.n += 1

        return self.n - 1, result

    def mean(self):
        arr_means = np.full((self.max, 1), np.inf)
        for key, group in self:
            arr_means[key - 1] = group.mean().values[0]
        result = pd.DataFrame(arr_means, index=self.days_of_year)
        return result
