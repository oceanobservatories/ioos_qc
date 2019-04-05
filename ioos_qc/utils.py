#!/usr/bin/env python
# coding=utf-8
import os
import geojson
import logging
from pathlib import Path
from typing import Union
from numbers import Real
from datetime import datetime, date

import numpy as np
import netCDF4 as nc4

N = Real
L = logging.getLogger(__name__)  # noqa


def isfixedlength(lst : Union[list, tuple],
                  length : int
                  ) -> bool:
    if not isinstance(lst, (list, tuple)):
        raise ValueError('Required: list/tuple, Got: {}'.format(type(lst)))

    if len(lst) != length:
        raise ValueError(
            'Incorrect list/tuple length for {}. Required: {}, Got: {}'.format(
                lst,
                length,
                len(lst)
            )
        )

    return True


def isnan(v):
    return (
        v is None or
        v is np.nan or
        v is np.ma.masked
    )


def check_timestamps(times : np.ndarray,
                     max_time_interval : N = None):
    """Sanity checks for timestamp arrays

    Checks that the times supplied are in monotonically increasing
    chronological order, and optionally that time intervals between
    measurements do not exceed a value `max_time_interval`.  Note that this is
    not a QARTOD test, but rather a utility test to make sure times are in the
    proper order and optionally do not have large gaps prior to processing the
    data.

    Args:
        times: Input array of timestamps
        max_time_interval: The interval between values should not exceed this
            value. [optional]
    """

    time_diff = np.diff(times)
    sort_diff = np.diff(sorted(times))
    # Check if there are differences between sorted and unsorted, and then
    # see if if there are any duplicate times.  Then check that none of the
    # diffs exceeds the sorted time.
    zero = np.array(0, dtype=time_diff.dtype)
    if not np.array_equal(time_diff, sort_diff) or np.any(sort_diff == zero):
        return False
    elif (max_time_interval is not None and
          np.any(sort_diff > max_time_interval)):
        return False
    else:
        return True


def dict_update(d, u):
    # http://stackoverflow.com/a/3233356
    import collections
    for k, v in u.items():
        if isinstance(d, collections.Mapping):
            if isinstance(v, collections.Mapping):
                r = dict_update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k] }
    return d


def cf_safe_name(name):
    import re
    if isinstance(name, str):
        if re.match('^[0-9_]', name):
            # Add a letter to the front
            name = "v_{}".format(name)
        return re.sub(r'[^_a-zA-Z0-9]', "_", name)

    raise ValueError('Could not convert "{}" to a safe name'.format(name))


class GeoNumpyDateEncoder(geojson.GeoJSONEncoder):

    def default(self, obj):
        """If input object is an ndarray it will be converted into a list
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return np.asscalar(obj)
        # elif isinstance(obj, pd.Timestamp):
        #     return obj.to_pydatetime().isoformat()
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif np.isnan(obj):
            return None

        return geojson.factory.GeoJSON.to_instance(obj)


def ncd_from_object(path_or_ncd, create=True, mode=None):

    created = False
    mode = mode or 'a'

    if isinstance(path_or_ncd, (str, Path)):

        if os.path.exists(str(path_or_ncd)):
            try:
                ncd = nc4.Dataset(str(path_or_ncd), mode)
            except OSError:
                # Create new file
                if create is True:
                    ncd = nc4.Dataset(str(path_or_ncd), 'w')
                    created = True
                else:
                    raise ValueError('Input is not an existing file path or Dataset')
        else:
            # Create new file
            if create is True:
                ncd = nc4.Dataset(str(path_or_ncd), 'w')
                created = True
            else:
                raise ValueError('Input is not a existing file path or Dataset')

    elif isinstance(path_or_ncd, nc4.Dataset):
        ncd = path_or_ncd

    else:
        raise ValueError('Input is not a valid file path or netCDF4 Dataset object')

    return ncd, created
