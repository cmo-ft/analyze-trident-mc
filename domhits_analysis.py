import numpy as np
import pandas as pd
import os
import time
import copy
import glob
import math

time_window = 20 #ns


def get_coincidence_level_with_showerId(pmthits, time_window: float=time_window):
    """
    get coincidence level with showerId as index. Each shower is expected to be short time
    :param pmthits:
    :param time_window:
    :return:
    """
    def get_coincidence_level_df(pmthits):
        """
        :param dom_hits: a dataframe whose index is ['DomId'] and contains all pmt hits
        :return:
        """
        minT = pmthits['t0'].min()
        maxT = minT + time_window  # time window
        dom_hits = pmthits.loc[pmthits.t0 < maxT]
        return len(dom_hits.groupby('PmtId'))

    coincidence_level = pmthits.groupby(['showerId','DomId']).apply(get_coincidence_level_df)
    dom_hits = pmthits.groupby(['showerId','DomId']).min()
    dom_hits['coincidence_level'] = coincidence_level
    coincidence_level = dom_hits[['weight','coincidence_level']].groupby(['DomId','coincidence_level']).sum()
    new_index = pd.MultiIndex.from_product([coincidence_level.index.get_level_values(0).unique(), range(1, 32)], names=['DomId', 'coincidence_level'])
    coincidence_level = coincidence_level.reindex(new_index)
    coincidence_level = coincidence_level.fillna(0)
    return dom_hits, coincidence_level


def get_coincidence_level_time_range(pmthits: pd.DataFrame, time_window: float=time_window):
    """
    get coincidence level for a long range time of hits
    :param dom_hits: a dataframe contains all pmt hits
    :return:
    """
    dom_hits = pmthits.set_index('DomId')
    coincidence_level_for_each_dom = []
    coincidence_level_list = list(range(1, 32))

    for idom in dom_hits.index.unique():
        hits = dom_hits.loc[idom].copy().reset_index()
        maxT = hits.t0.max()

        hits['tid'] = (hits.t0 / time_window).astype(int)

        # manage hits at boundary of time window
        # Find the minimum time for hits in the time window, then find all hits with
        # t-tmin<time window in the nearby window
        tid = hits.tid.to_numpy()
        # print(np.where( np.diff(tid)==1 )[0])
        for idx in np.where( np.diff(tid)==1 )[0]:
            tmp_tid = tid[idx]
            tmin = np.inf
            rev_idx = 0
            while rev_idx<idx and \
                    hits.loc[idx-rev_idx].tid==tmp_tid:
                tmin = hits.loc[idx-rev_idx].t0
                rev_idx += 1
            tmax = tmin + time_window
            fwd_idx = 1
            while idx+fwd_idx<len(hits) and \
                    (hits.loc[idx+fwd_idx].tid==tmp_tid+1) and \
                     (hits.loc[idx+fwd_idx].t0<tmax):
                hits.loc[idx+fwd_idx, 'tid'] = tmp_tid

        hits = hits.groupby(['tid', 'PmtId']).first()
        nhits = hits.index.get_level_values(0).value_counts()
        coincidence_level_count = nhits.value_counts()
        coincidence_level_count = coincidence_level_count.reindex(pd.Index(coincidence_level_list)).fillna(0)

        coincidence_level_for_each_dom.append(
            pd.DataFrame({'DomId':idom,
                          'coincidence_level':coincidence_level_count.index,
                          'coincidence_level_count':coincidence_level_count,
                          'maxT': maxT
                          })
        )

    coincidence_level_for_each_dom = pd.concat(coincidence_level_for_each_dom)
    return coincidence_level_for_each_dom
