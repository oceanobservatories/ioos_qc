#!/usr/bin/env python
# coding=utf-8
import os
import logging
from pathlib import Path
import simplejson as json
from copy import deepcopy
from inspect import signature
from collections import OrderedDict
from importlib import import_module

import numpy as np
import xarray as xr
import netCDF4 as nc4
from ruamel import yaml

from ioos_qc.utils import dict_update, cf_safe_name, GeoNumpyDateEncoder, ncd_from_object

L = logging.getLogger(__name__)  # noqa


class QcConfig(object):

    def __init__(self, path_or_dict):
        if isinstance(path_or_dict, OrderedDict):
            y = path_or_dict
        elif isinstance(path_or_dict, dict):
            y = OrderedDict(path_or_dict)
        elif isinstance(path_or_dict, str):
            with open(path_or_dict) as f:
                y = OrderedDict(yaml.load(f.read(), Loader=yaml.Loader))
        elif isinstance(path_or_dict, Path):
            with path_or_dict.open() as f:
                y = OrderedDict(yaml.load(f.read(), Loader=yaml.Loader))
        else:
            return ValueError('Input is not valid file path or YAMLObject')

        self.config = y

    def run(self, **passedkwargs):
        """ Runs the tests that are defined in the config object.
            Returns a dictionary of the results as defined by the config
        """
        results = OrderedDict()
        for modu, tests in self.config.items():
            try:
                testpackage = import_module('ioos_qc.{}'.format(modu))
            except ImportError:
                raise ValueError('No ioos_qc test package "{}" was found, skipping.'.format(modu))

            results[modu] = OrderedDict()
            for testname, kwargs in tests.items():
                if not hasattr(testpackage, testname):
                    L.warning('No test named "{}.{}" was found, skipping'.format(modu, testname))
                elif kwargs is None:
                    L.debug('Test "{}.{}" had no config, skipping'.format(modu, testname))
                else:
                    # Get our own copy of the kwargs object so we can change it
                    testkwargs = deepcopy(passedkwargs)
                    # Merges dicts
                    testkwargs = { **kwargs, **testkwargs }  # noqa

                    # Get the arguments that the test functions support
                    runfunc = getattr(testpackage, testname)
                    sig = signature(runfunc)
                    valid_keywords = [
                        p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD
                    ]

                    testkwargs = { k: v for k, v in testkwargs.items() if k in valid_keywords }
                    results[modu][testname] = runfunc(**testkwargs)  # noqa

        return results

    def __str__(self):
        """ A human friendly representation of the tests that this QcConfig object defines. """
        return str(self.config)


class NcQcConfig(QcConfig):

    def __init__(self, path_or_ncd_or_dict):

        load_as_dataset = False
        if isinstance(path_or_ncd_or_dict, OrderedDict):
            y = path_or_ncd_or_dict
        elif isinstance(path_or_ncd_or_dict, dict):
            y = OrderedDict(path_or_ncd_or_dict)
        elif isinstance(path_or_ncd_or_dict, str):
            try:
                with open(path_or_ncd_or_dict) as f:
                    y = OrderedDict(yaml.load(f.read(), Loader=yaml.Loader))
            except BaseException:
                load_as_dataset = True
        elif isinstance(path_or_ncd_or_dict, Path):
            try:
                with path_or_ncd_or_dict.open() as f:
                    y = OrderedDict(yaml.load(f.read(), Loader=yaml.Loader))
            except BaseException:
                load_as_dataset = True

        if load_as_dataset is True:
            y = OrderedDict()
            with xr.open_dataset(path_or_ncd_or_dict, decode_cf=False) as ds:
                ds = ds.filter_by_attrs(
                    ioos_qc_module=lambda x: x is not None,
                    ioos_qc_test=lambda x: x is not None,
                    ioos_qc_config=lambda x: x is not None,
                    ioos_qc_target=lambda x: x is not None,
                )
                for dv in ds.variables:
                    vobj = ds[dv]

                    # Because a data variables can have more than one check
                    # associated with it we need to merge any existing configs
                    # for this variable
                    newdict = OrderedDict({
                        vobj.ioos_qc_module: OrderedDict({
                            vobj.ioos_qc_test: OrderedDict(json.loads(vobj.ioos_qc_config))
                        })
                    })
                    merged = dict_update(
                        y.get(vobj.ioos_qc_target, {}),
                        newdict
                    )
                    y[vobj.ioos_qc_target] = merged

        self.config = y

    def run(self, path_or_ncd, **passedkwargs):
        """ Runs the tests that are defined in the config object.
            Returns a dictionary of the results as defined by the config
        """
        results = OrderedDict()

        with xr.open_dataset(path_or_ncd, decode_cf=False) as ds:
            for vname, qcobj in self.config.items():
                qc = QcConfig(qcobj)
                # Find any var specific kwargs to pass onto the run
                if vname not in ds.variables:
                    L.warning('{} not in Dataset, skipping'.format(vname))
                    continue

                varkwargs = { 'inp': ds.variables[vname].values }
                if vname in passedkwargs:
                    varkwargs = dict_update(varkwargs, passedkwargs[vname])

                results[vname] = qc.run(**varkwargs)
        return results

    def save_to_netcdf(self, path_or_ncd, results, data_source=None, modify_source=False):
        ncd = None
        dsncd = None

        try:
            # The file we are writing the quality results into
            # If the user passed in an open object we should not close it
            ncd, ncd_created = ncd_from_object(path_or_ncd, mode='a')
            # The data source, which will be equal to the `ncd` object unless we are writing
            # to a stand-alone file. We need to this to pull the correct dimension names and
            # sizes so the files can be merged back together easily. Don't create this if
            # it doesn't exist
            if data_source is not None:
                dsmode = 'r'
                if modify_source is True:
                    dsmode = 'a'
                dsncd, _ = ncd_from_object(data_source, create=False, mode=dsmode)
            else:
                dsncd = ncd

            num_qc_tests = 0  # A counter we use for naming auto-generated variables if we need to

            # Iterate over each variable
            for vname, qcobj in self.config.items():

                if vname not in dsncd.variables:
                    L.warning('{} not found in the Dataset, skipping'.format(vname))
                    continue
                source_var = dsncd.variables[vname]


                if vname not in results:
                    L.warning('No results for {}, skipping'.format(vname))
                    continue

                # Iterate over each module
                for modu, tests in qcobj.items():

                    if modu not in results[vname]:
                        L.warning('No results for {}.{}, skipping'.format(vname, modu))
                        continue

                    try:
                        testpackage = import_module('ioos_qc.{}'.format(modu))
                    except ImportError:
                        L.error('No ioos_qc test package "{}" was found, skipping.'.format(modu))
                        continue

                    # Keep track of the test names so we can add to the source's
                    # ancillary_variables at the end
                    qcvar_names = []
                    for testname, kwargs in tests.items():

                        if testname not in results[vname][modu]:
                            L.warning('No results for {}.{}.{}, skipping'.format(
                                vname, modu, testname
                            ))
                            continue

                        # Try to find a qc variable that matches this config
                        qcvars = ncd.get_variables_by_attributes(
                            ioos_qc_module=modu,
                            ioos_qc_test=testname,
                            ioos_qc_target=vname
                        )
                        if not qcvars:
                            # Generate a random variable name for this check. This is stored in the
                            # targets `ancillary_variables`.
                            qcvarname = "{0}.{1}.{2}".format(vname, modu, testname)
                            qcvarname = "{0}_{1:03d}".format(qcvarname, num_qc_tests)
                            qcvarname = cf_safe_name(qcvarname)
                            # Iterate the number of qc tests so we can not overwrite previously
                            # created variables
                            num_qc_tests += 1
                        else:
                            if len(qcvars) > 1:
                                names = [ v.name for v in qcvars ]
                                L.warning('Found more than one QC variable match: {}'.format(names))
                            # Use the last one found
                            qcvarname = qcvars[-1].name

                        varresults = np.asarray(results[vname][modu][testname])
                        varconfig = json.dumps(
                            kwargs,
                            cls=GeoNumpyDateEncoder,
                            allow_nan=False,
                            ignore_nan=True
                        )

                        # Get flags from module attribute called FLAGS
                        flags = getattr(testpackage, 'FLAGS')
                        varflagnames = [ d for d in flags.__dict__ if not d.startswith('__') ]
                        varflagvalues = [ getattr(flags, d) for d in varflagnames ]

                        for d in source_var.get_dims():
                            if d.name not in ncd.dimensions:
                                ncd.createDimension(d.name, size=d.size)

                        if qcvarname not in ncd.variables:
                            v = ncd.createVariable(qcvarname, np.byte, source_var.dimensions)
                        else:
                            v = ncd[qcvarname]

                        v.setncattr('standard_name', 'status_flag')
                        v.setncattr('flag_values', np.byte(varflagvalues))
                        v.setncattr('flag_meanings', ' '.join(varflagnames))
                        v.setncattr('valid_min', np.byte(min(varflagvalues)))
                        v.setncattr('valid_max', np.byte(max(varflagvalues)))
                        v.setncattr('ioos_qc_config', varconfig)
                        v.setncattr('ioos_qc_module', modu)
                        v.setncattr('ioos_qc_test', testname)
                        v.setncattr('ioos_qc_target', vname)
                        v[:] = varresults

                        qcvar_names.append(qcvarname)

                # Update the source ancillary_variables if we are appending to an existing file
                if modify_source:
                    existing = getattr(source_var, 'ancillary_variables', '').split(' ')
                    existing += qcvar_names
                    source_var.ancillary_variables = ' '.join(list(set(existing))).strip()

        finally:
            if dsncd is not None:
                dsncd.close()

            if data_source is not None and ncd is not None:
                ncd.close()

    def __getattr__(self, item):
        if item in self.config:
            return self.config[item]
        else:
            return self.__getattribute__(item)

    def __getitem__(self, item):
        if item in self.config:
            return self.config[item]
        else:
            raise KeyError('{} not found in config object'.format(item))

    def __str__(self):
        """ A human friendly representation of the tests that this QcConfig object defines. """
        return json.dumps(
            self.config,
            cls=GeoNumpyDateEncoder,
            allow_nan=False,
            ignore_nan=True,
            indent=2,
        )

    def __repr__(self):
        """ A human friendly representation of the tests that this QcConfig object defines. """
        return str(self)
