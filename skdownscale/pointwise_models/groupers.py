import warnings

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
        self.n = 1
        self.df = df
        self.max = 366
        # check for leap days
        # if leap days present, flag for day groups count
        if len(self.df[((self.df.index.month == 2) & (self.df.index.day == 29))]) > 0:
            self.leap = 'leap'
        else:
            self.leap = 'noleap'
        # split up data by leap and non leap years
        # necessary because pandas dayofyear
        self.df_leap = self.df[self.df.index.is_leap_year]
        self.df_noleap = self.df[~self.df.index.is_leap_year]
        self.offset = offset
        self.days_of_nonleap_year = np.arange(self.n, self.max)
        self.days_of_leap_year = np.arange(self.n, self.max + 1)
        self.days_of_nonleap_year_wrapped = np.pad(
            self.days_of_nonleap_year, self.offset, mode='wrap'
        )
        self.days_of_leap_year_wrapped = np.pad(self.days_of_leap_year, self.offset, mode='wrap')

    def __iter__(self):
        self.n = 1
        return self

    def __next__(self):
        # n as day of year
        if self.n > self.max:
            raise StopIteration

        i = self.n - 1
        total_days = (2 * self.offset) + 1

        # create day groups with +/- offset # of days
        first_set_leap = self.days_of_leap_year_wrapped[i : i + self.offset]
        first_set_noleap = self.days_of_nonleap_year_wrapped[i : i + self.offset]

        sec_set_leap = self.days_of_leap_year_wrapped[self.n + self.offset : i + total_days]
        sec_set_noleap = self.days_of_nonleap_year_wrapped[self.n + self.offset : i + total_days]

        all_days_leap = np.concatenate((first_set_leap, np.array([self.n]), sec_set_leap), axis=0)
        all_days_noleap = np.concatenate(
            (first_set_noleap, np.array([self.n]), sec_set_noleap), axis=0
        )

        # check that day groups contain the correct number of days
        if len(set(all_days_leap)) != total_days and self.leap == 'noleap':
            warnings.warn('leap days not included, day groups in leap years missing leap days')

        if len(set(all_days_noleap)) != total_days and self.n != 366:
            raise ValueError('no leap day groups do not contain the correct set of days')

        result = pd.concat(
            [
                self.df_leap[self.df_leap.index.dayofyear.isin(all_days_leap)],
                self.df_noleap[self.df_noleap.index.dayofyear.isin(all_days_noleap)],
            ]
        )

        self.n += 1

        return self.n - 1, result

    def mean(self):
        arr_means = np.full((self.max, 1), np.inf)
        for key, group in self:
            arr_means[key - 1] = group.mean().values[0]
        result = pd.DataFrame(arr_means, index=self.days_of_leap_year)
        return result
