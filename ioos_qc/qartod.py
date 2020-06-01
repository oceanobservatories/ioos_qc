#!/usr/bin/env python
# coding=utf-8
import logging
import warnings
from collections import namedtuple
from numbers import Real as N

import numpy as np
import pandas as pd
from pygc import great_distance

from ioos_qc.utils import (
    isnan,
    isfixedlength
)

L = logging.getLogger(__name__)  # noqa


class QartodFlags(object):
    """Primary flags for QARTOD."""

    GOOD = 1
    UNKNOWN = 2
    SUSPECT = 3
    FAIL = 4
    MISSING = 9


FLAGS = QartodFlags  # Default name for all check modules

span = namedtuple('Span', 'minv maxv')


def mapdates(dates):
    if hasattr(dates, 'dtype') and np.issubdtype(dates.dtype, np.datetime64):
        # numpy datetime objects
        return dates.astype('datetime64[ns]')
    else:
        try:
            # Finally try unix epoch seconds
            return pd.to_datetime(dates, unit='s').values.astype('datetime64[ns]')
        except Exception:
            # strings work here but we don't advertise that
            return np.array(dates, dtype='datetime64[ns]')
            

def qartod_compare(vectors):
    """Aggregates an array of flags by precedence into a single array.

    Args:
        vectors: An array of uniform length arrays representing individual flags

    Returns:
        A masked array of aggregated flag data.
    """
    shapes = [v.shape[0] for v in vectors]
    # Assert that all of the vectors are the same size.
    assert all([s == shapes[0] for s in shapes])
    assert all([v.ndim == 1 for v in vectors])

    result = np.ma.empty(shapes[0])
    result.fill(QartodFlags.MISSING)

    priorities = [
        QartodFlags.MISSING,
        QartodFlags.UNKNOWN,
        QartodFlags.GOOD,
        QartodFlags.SUSPECT,
        QartodFlags.FAIL
    ]
    # For each of the priorities in order, set the resultant array to the the
    # flag where that flag exists in each of the vectors.
    for p in priorities:
        for v in vectors:
            idx = np.where(v == p)[0]
            result[idx] = p
    return result


def location_test(lon, lat, bbox = (-180, -90, 180, 90), range_max = None):
    """Checks that a location is within reasonable bounds.

    Checks that longitude and latitude are within reasonable bounds defaulting
    to lon = [-180, 180] and lat = [-90, 90].  Optionally, check for a maximum
    range parameter in great circle distance defaulting to meters which can
    also use a unit from the quantities library. Missing and masked data is
    flagged as UNKNOWN.

    Args:
        lon: Longitudes as a numeric numpy array or a list of numbers.
        lat: Latitudes as a numeric numpy array or a list of numbers.
        bbox: A length 4 tuple expressed in (minx, miny, maxx, maxy) [optional].
        range_max: Maximum allowed range expressed in geodesic curve distance (meters).

    Returns:
        A masked array of flag values equal in size to that of the input.
    """

    bboxnt = namedtuple('BBOX', 'minx miny maxx maxy')
    if bbox is not None:
        assert isfixedlength(bbox, 4)
        bbox = bboxnt(*bbox)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat = np.ma.masked_invalid(np.array(lat).astype(np.floating))
        lon = np.ma.masked_invalid(np.array(lon).astype(np.floating))

    if lon.shape != lat.shape:
        raise ValueError(
            'Lon ({0.shape}) and lat ({1.shape}) are different shapes'.format(
                lon, lat
            )
        )

    # Save original shape
    original_shape = lon.shape
    lon = lon.flatten()
    lat = lat.flatten()

    # Start with everything as passing (1)
    flag_arr = np.ma.ones(lon.size, dtype='uint8')

    # If either lon or lat are masked we just set the flag to MISSING
    mloc = lon.mask & lat.mask
    flag_arr[mloc] = QartodFlags.MISSING

    # If there is only one masked value fail the location test
    mismatch = lon.mask != lat.mask
    flag_arr[mismatch] = QartodFlags.FAIL

    if range_max is not None and lon.size > 1:
        # Calculating the great_distance between each point
        # Flag suspect any distance over range_max
        d = np.ma.zeros(lon.size, dtype=np.float64)
        d[1:] = great_distance(
            start_latitude=lat[:-1],
            end_latitude=lat[1:],
            start_longitude=lon[:-1],
            end_longitude=lon[1:]
        )['distance']
        flag_arr[d > range_max] = QartodFlags.SUSPECT

    # Ignore warnings when comparing NaN values even though they are masked
    # https://github.com/numpy/numpy/blob/master/doc/release/1.8.0-notes.rst#runtime-warnings-when-comparing-nan-numbers
    with np.errstate(invalid='ignore'):
        flag_arr[(lon < bbox.minx) | (lat < bbox.miny) |
                 (lon > bbox.maxx) | (lat > bbox.maxy)] = QartodFlags.FAIL

    return flag_arr.reshape(original_shape)


def gross_range_test(inp, fail_span, suspect_span = None):
    """Checks that values are within reasonable range bounds.

    Given a 2-tuple of minimum/maximum values, flag data outside of the given
    range as FAIL data.  Optionally also flag data which falls outside of a user
    defined range as SUSPECT. Missing and masked data is flagged as UNKNOWN.

    Args:
        inp: Input data as a numeric numpy array or a list of numbers.
        fail_span: 2-tuple range which to flag outside data as FAIL.
        suspect_span: 2-tuple range which to flag outside data as SUSPECT. [optional]

    Returns:
        A masked array of flag values equal in size to that of the input.
    """

    assert isfixedlength(fail_span, 2)
    sspan = span(*sorted(fail_span))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inp = np.ma.masked_invalid(np.array(inp).astype(np.floating))

    # Save original shape
    original_shape = inp.shape
    inp = inp.flatten()
    # Start with everything as passing (1)
    flag_arr = np.ma.ones(inp.size, dtype='uint8')

    # If the value is masked set the flag to MISSING
    flag_arr[inp.mask] = QartodFlags.MISSING

    if suspect_span is not None:
        assert isfixedlength(suspect_span, 2)
        uspan = span(*sorted(suspect_span))
        if uspan.minv < sspan.minv or uspan.maxv > sspan.maxv:
            raise ValueError('User span range may not exceed sensor span')
        # Flag suspect outside of user span
        with np.errstate(invalid='ignore'):
            flag_arr[(inp < uspan.minv) | (inp > uspan.maxv)] = QartodFlags.SUSPECT

    # Flag suspect outside of sensor span
    with np.errstate(invalid='ignore'):
        flag_arr[(inp < sspan.minv) | (inp > sspan.maxv)] = QartodFlags.FAIL

    return flag_arr.reshape(original_shape)


class ClimatologyConfig(object):
    """
    Args:
        tspan: 2-tuple range.
                If period is defined, then this is a numeric range.
                If period is not defined, then its a date range.
        fspan: (optional) 2-tuple range of valid values. This is passed in as the fail_span to the gross_range test.
        vspan: 2-tuple range of valid values. This is passed in as the suspect_span to the gross_range test.
        zspan: (optional) Vertical (depth) range, in meters positive down
        period: (optional) The unit the tspan argument is in. Defaults to datetime object
                but can also be any attribute supported by a pandas Timestamp object.
                See: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html                    
                    * year
                    * week / weekofyear
                    * dayofyear
                    * dayofweek
                    * quarter
    """
    mem = namedtuple('window', [
        'tspan',
        'fspan',
        'vspan',
        'zspan',
        'period'
    ])

    def __init__(self, members=None):
        members = members or []
        self._members = members

    @property
    def members(self):
        return self._members

    def values(self, tind, zind=None):
        """
        Args:
            tind: Value to test for inclusion between time bounds
        """
        span = (None, None)
        for m in self._members:

            if m.period is not None:
                # If a period is defined, extract the attribute from the
                # pd.Timestamp object before comparison. The min and max
                # values are in this period unit already.
                tind_copy = getattr(tind, m.period)
            else:
                # If a period isn't defined, make a new Timestamp object
                # to align with the above name 'tind_copy'
                tind_copy = tind

            # If we are between times
            if tind_copy > m.tspan.minv and tind_copy <= m.tspan.maxv:
                if not isnan(zind) and not isnan(m.zspan):
                    # If we are between depths
                    if zind > m.zspan.minv and zind <= m.zspan.maxv:
                        span = m.vspan
                elif isnan(zind) and isnan(m.zspan):
                    span = m.vspan
        return span

    def add(self, tspan, vspan, fspan = None, zspan = None, period = None):

        assert isfixedlength(tspan, 2)
        # If period is defined, tspan is a numeric
        # if it isn't defined, its a parsable date
        if period is not None:
            tspan = span(*sorted(tspan))
        else:
            tspan = span(*sorted([
                pd.Timestamp(tspan[0]),
                pd.Timestamp(tspan[1])
            ]))

        assert isfixedlength(vspan, 2)
        vspan = span(*sorted(vspan))

        if fspan is not None:
            assert isfixedlength(fspan, 2)
            fspan = span(*sorted(fspan))

        if zspan is not None:
            assert isfixedlength(zspan, 2)
            zspan = span(*sorted(zspan))

        if period is not None:
            # Test to make sure this is callable on a Timestamp
            try:
                getattr(pd.Timestamp.now(), period)
            except AttributeError:
                raise ValueError('The period "{period}" is not recognized')

        self._members.append(
            self.mem(
                tspan,
                fspan,
                vspan,
                zspan,
                period
            )
        )

    def check(self, tinp, inp, zinp):
        # Start with everything as UNKNOWN (1)
        flag_arr = np.ma.empty(inp.size, dtype='uint8')
        flag_arr.fill(QartodFlags.UNKNOWN)

        # If the value is masked set the flag to MISSING
        flag_arr[inp.mask] = QartodFlags.MISSING

        # Iterate over each member and apply its spans on the input data.
        # Member spans are applied in order and any data points that fall into
        # more than one member are flagged by each one.
        for m in self._members:
            
            if m.period is not None:
                # If a period is defined, extract the attribute from the
                # pd.DatetimeIndex object before comparison. The min and max
                # values are in this period unit already.
                tinp_copy = getattr(tinp, m.period).to_series()
            else:
                # If a period isn't defined, make a new Timestamp object
                # to align with the above name 'tinp_copy'
                tinp_copy = tinp

            # If a zspan is defined but we don't have z input (zinp), skip this member
            # Note: `zinp.count()` can return `np.ma.masked` so we also check using isnan
            if not isnan(m.zspan) and (not zinp.count() or isnan(zinp.any())):
                continue

            # Indexes that align with the T
            t_idx = (tinp_copy >= m.tspan.minv) & (tinp_copy <= m.tspan.maxv)

            # Indexes that align with the Z
            if not isnan(m.zspan):
                # Only test non-masked values between the min and max
                z_idx = (~zinp.mask) & (zinp >= m.zspan.minv) & (zinp <= m.zspan.maxv)
            else:
                # If there is no z data in the config, don't try to filter by depth!
                # Set z_idx to all True to prevent filtering
                z_idx = np.ones(inp.size, dtype=bool)
                
            # Combine the T and Z indexes
            values_idx = (t_idx & z_idx)

            # Failed and suspect data for this value span. Combining fail_idx or
            # suspect_idx with values_idx represents the subsets of data that should be
            # fail and suspect respectively.
            if not isnan(m.fspan):
                fail_idx = (inp < m.fspan.minv) | (inp > m.fspan.maxv)
            else:
                fail_idx = np.zeros(inp.size, dtype=bool)

            suspect_idx = (inp < m.vspan.minv) | (inp > m.vspan.maxv)
            
            with np.errstate(invalid='ignore'):
                flag_arr[(values_idx & fail_idx)] = QartodFlags.FAIL
                flag_arr[(values_idx & ~fail_idx & suspect_idx)] = QartodFlags.SUSPECT
                flag_arr[(values_idx & ~fail_idx & ~suspect_idx)] = QartodFlags.GOOD

        return flag_arr

    @staticmethod
    def convert(config):
        # Create a ClimatologyConfig object if one was not passed in
        if isinstance(config, ClimatologyConfig):
            return config

        c = ClimatologyConfig()
        for climate_config_dict in config:
            c.add(**climate_config_dict)
        return c


def climatology_test(config, inp, tinp, zinp=None):
    """Checks that values are within reasonable range bounds and flags as SUSPECT.

    Data for which no ClimatologyConfig member exists is marked as UNKNOWN.

    Args:
        config: A ClimatologyConfig object or a list of dicts containing tuples
            that can be used to create a ClimatologyConfig object. See ClimatologyConfig
            docs for more info.
        tinp: Time data as a sequence of datetime objects compatible with pandas DatetimeIndex.
          This includes numpy datetime64, python datetime objects and pandas Timestamp object.
          ie. pd.DatetimeIndex([datetime.utcnow(), np.datetime64(), pd.Timestamp.now()]
          If anything else is passed in the format is assumed to be seconds since the unix epoch.
        vinp: Input data as a numeric numpy array or a list of numbers.
        zinp: Z (depth) data, in meters positive down, as a numeric numpy array or a list of numbers.

    Returns:
        A masked array of flag values equal in size to that of the input.
    """

    # Create a ClimatologyConfig object if one was not passed in
    config = ClimatologyConfig.convert(config)

    tinp = mapdates(tinp)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inp = np.ma.masked_invalid(np.array(inp).astype(np.floating))
        zinp = np.ma.masked_invalid(np.array(zinp).astype(np.floating))

    # Save original shape
    original_shape = inp.shape

    # We compare using a pandas Timestamp for helper functions like
    # 'week' and 'dayofyear'. It is surprisingly hard to pull these out
    # of a plain datetime64 object.
    tinp = pd.DatetimeIndex(tinp.flatten())
    inp = inp.flatten()
    zinp = zinp.flatten()

    flag_arr = config.check(tinp, inp, zinp)
    return flag_arr.reshape(original_shape)


def spike_test(inp, suspect_threshold, fail_threshold):
    """Check for spikes by checking neighboring data against thresholds

    Determine if there is a spike at data point n-1 by subtracting
    the midpoint of n and n-2 and taking the absolute value of this
    quantity, and checking if it exceeds a low or high threshold.
    Values which do not exceed either threshold are flagged GOOD,
    values which exceed the low threshold are flagged SUSPECT,
    and values which exceed the high threshold are flagged FAIL.
    Missing and masked data is flagged as UNKNOWN.

    Args:
        inp: Input data as a numeric numpy array or a list of numbers.
        suspect_threshold: The SUSPECT threshold value, in observations units.
        fail_threshold: The SUSPECT threshold value, in observations units.

    Returns:
        A masked array of flag values equal in size to that of the input.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inp = np.ma.masked_invalid(np.array(inp).astype(np.floating))

    # Save original shape
    original_shape = inp.shape
    inp = inp.flatten()

    # Calculate the average of n-2 and n
    ref = np.zeros(inp.size, dtype=np.float64)
    ref[1:-1] = (inp[0:-2] + inp[2:]) / 2
    ref = np.ma.masked_invalid(ref)

    # Start with everything as passing (1)
    flag_arr = np.ma.ones(inp.size, dtype='uint8')

    # Calculate the (n-1 - ref) difference
    diff = np.abs(inp - ref)

    # If n-1 - ref is greater than the low threshold, SUSPECT test
    with np.errstate(invalid='ignore'):
        flag_arr[diff > suspect_threshold] = QartodFlags.SUSPECT

    # If n-1 - ref is greater than the high threshold, FAIL test
    with np.errstate(invalid='ignore'):
        flag_arr[diff > fail_threshold] = QartodFlags.FAIL

    # If the value is masked or nan set the flag to MISSING
    flag_arr[diff.mask] = QartodFlags.MISSING

    return flag_arr.reshape(original_shape)


def rate_of_change_test(inp, tinp, threshold):
    """Checks the first order difference of a series of values to see if
    there are any values exceeding a threshold defined by the inputs.
    These are then marked as SUSPECT.  It is up to the test operator
    to determine an appropriate threshold value for the absolute difference not to
    exceed. Threshold is expressed as a rate in observations units per second.
    Missing and masked data is flagged as UNKNOWN.

    Args:
        inp: Input data as a numeric numpy array or a list of numbers.
        tinp: Time data as a sequence of datetime objects compatible with pandas DatetimeIndex.
              This includes numpy datetime64, python datetime objects and pandas Timestamp object.
              ie. pd.DatetimeIndex([datetime.utcnow(), np.datetime64(), pd.Timestamp.now()]
              If anything else is passed in the format is assumed to be seconds since the unix epoch.
        threshold: A float value representing a rate of change over time,
                   in observation units per second.

    Returns:
        A masked array of flag values equal in size to that of the input.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inp = np.ma.masked_invalid(np.array(inp).astype(np.floating))

    # Save original shape
    original_shape = inp.shape
    inp = inp.flatten()

    # Start with everything as passing (1)
    flag_arr = np.ma.ones(inp.size, dtype='uint8')

    # calculate rate of change in units/second
    roc = np.ma.zeros(inp.size, dtype='float')

    tinp = mapdates(tinp).flatten()
    roc[1:] = np.abs(np.diff(inp) / np.diff(tinp).astype('timedelta64[s]').astype(float))

    with np.errstate(invalid='ignore'):
        flag_arr[roc > threshold] = QartodFlags.SUSPECT

    # If the value is masked set the flag to MISSING
    flag_arr[inp.mask] = QartodFlags.MISSING

    return flag_arr.reshape(original_shape)


def flat_line_test(inp, tinp, suspect_threshold, fail_threshold, tolerance = 0):
    """Check for consecutively repeated values within a tolerance.
    Missing and masked data is flagged as UNKNOWN.
    More information: https://github.com/ioos/ioos_qc/pull/11
    Args:
        inp: Input data as a numeric numpy array or a list of numbers.
        tinp: Time data as a sequence of datetime objects compatible with pandas DatetimeIndex.
              This includes numpy datetime64, python datetime objects and pandas Timestamp object.
              ie. pd.DatetimeIndex([datetime.utcnow(), np.datetime64(), pd.Timestamp.now()]
              If anything else is passed in the format is assumed to be seconds since the unix epoch.
        suspect_threshold: The number of seconds within `tolerance` to
            allow before being flagged as SUSPECT.
        fail_threshold: The number of seconds within `tolerance` to
            allow before being flagged as FAIL.
        tolerance: The tolerance that should be exceeded between consecutive values.
            To determine if the current point `n` should be flagged, we use a rolling window, with endpoint at
            point `n`, and calculate the range of values in the window. If that range is less than `tolerance`,
            then the point is flagged.
    Returns:
        A masked array of flag values equal in size to that of the input.
    """

    # input as numpy arr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inp = np.ma.masked_invalid(np.array(inp).astype(np.floating))

    # Save original shape
    original_shape = inp.shape
    inp = inp.flatten()

    # Start with everything as passing
    flag_arr = np.full((inp.size,), QartodFlags.GOOD)

    # if we have fewer than 3 points, we can't run the test, so everything passes
    if len(inp) < 3:
        return flag_arr.reshape(original_shape)

    # determine median time interval
    tinp = mapdates(tinp).flatten()

    # The thresholds are in seconds so we round make sure the interval is also in seconds
    time_interval = np.median(np.diff(tinp)).astype('timedelta64[s]').astype(float)

    def rolling_window(a, window):
        """
        https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
        """
        if len(a) < window:
            return np.ma.MaskedArray(np.empty((0, window + 1)))
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window + 1)
        strides = a.strides + (a.strides[-1],)
        arr = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        return np.ma.masked_invalid(arr[:-1, :])

    def run_test(test_threshold, flag_value):
        # convert time thresholds to number of observations
        count = (int(test_threshold) / time_interval).astype(int)

        # calculate actual data ranges for each window
        window = rolling_window(inp, count)
        data_min = np.min(window, 1)
        data_max = np.max(window, 1)
        data_range = np.abs(data_max - data_min)

        # find data ranges that are within threshold and flag them
        test_results = np.ma.filled(data_range < tolerance, fill_value=False)
        # data points before end of first window should pass
        n_fill = count if count < len(inp) else len(inp)
        test_results = np.insert(test_results, 0, np.full((n_fill,), False))
        flag_arr[test_results] = flag_value

    run_test(suspect_threshold, QartodFlags.SUSPECT)
    run_test(fail_threshold, QartodFlags.FAIL)

    # If the value is masked set the flag to MISSING
    flag_arr[inp.mask] = QartodFlags.MISSING

    return flag_arr.reshape(original_shape)


def attenuated_signal_test(inp, threshold, check_type = 'std'):
    """Check for near-flat-line conditions using a range or standard deviation.

    Missing and masked data is flagged as UNKNOWN.

    Args:
        inp: Input data as a numeric numpy array or a list of numbers.
        threshold: 2-tuple representing the minimum thresholds to use for SUSPECT
            and FAIL checks. The smaller of the two values is used in the SUSPECT
            tests and the greater of the two values is used in the FAIL tests.
        check_type: Either 'std' (default) or 'range', depending on the type of check
            you wish to perform.

    Returns:
        A masked array of flag values equal in size to that of the input.
        This array will always contain only a single unique value since all
        input data is flagged together.
    """

    assert isfixedlength(threshold, 2)
    threshold = span(*reversed(sorted(threshold)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inp = np.ma.masked_invalid(np.array(inp).astype(np.floating))

    # Save original shape
    original_shape = inp.shape
    inp = inp.flatten()

    if check_type == 'std':
        check_val = np.std(inp)
    elif check_type == 'range':
        check_val = np.ptp(inp)
    else:
        raise ValueError('Check type "{}" is not defined'.format(check_type))

    # Start with everything as passing (1)
    flag_arr = np.ma.ones(inp.size, dtype='uint8')

    if check_val < threshold.maxv:
        flag_arr.fill(QartodFlags.FAIL)
    elif check_val < threshold.minv:
        flag_arr.fill(QartodFlags.SUSPECT)

    # If the value is masked set the flag to MISSING
    flag_arr[inp.mask] = QartodFlags.MISSING

    return flag_arr.reshape(original_shape)
