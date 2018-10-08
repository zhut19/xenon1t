from datetime import datetime

from hax.minitrees import TreeMaker
from hax.paxroot import loop_over_dataset
from hax.utils import find_file_in_folders, get_user_id
from hax.minitree_formats import get_format
from hax import runs
import hax

import pickle

import numpy as np
import pandas as pd


class TwoLevelTreeMaker(TreeMaker):
    EventLevelTreeMaker = None

    def __init__(self):
        # Support for string arguments
        if isinstance(self.branch_selection, str):
            self.branch_selection = [self.branch_selection]
        if isinstance(self.extra_branches, str):
            self.extra_branches = [self.extra_branches]

        if not self.branch_selection:
            self.branch_selection = hax.config['basic_branches'] + \
                list(self.extra_branches)
        if 'event_number' not in self.branch_selection:
            self.branch_selection += ['event_number']

        self.cache = dict(peak=[], event=[])  # Peak level cache
        self.data = []  # Peak level data
        self.data_event = []  # Event level data

    # Logistics that works for two level treemaker
    def process_event(self, event):
        result_peaks, result_events = self.extract_data(event)
        if not isinstance(result_peaks, (list, tuple)):
            raise TypeError("MultipleRowExtractor treemakers must extract "
                            "a list of dictionaries, not a %s" % type(result_peaks))
        if not isinstance(result_events, dict):
            raise ValueError("Event level treeMaker must extract dictionary")

        self.cache['peak'].extend(result_peaks)
        self.cache['event'].append(result_events)

    def check_cache(self, force_empty=False):
        for level in ['peak', 'event']:
            if not len(self.cache[level]) \
                or (len(self.cache[level]) < self.cache_size
                    and not force_empty):
                return

            if level == 'peak':
                self.data.append(pd.DataFrame(self.cache[level]))
            elif level == 'event':
                self.data_event.append(pd.DataFrame(self.cache[level]))

            self.cache[level] = []

    def get_data(self, dataset, event_list=None):
        """Return data extracted from running over dataset"""
        self.mc_data = runs.is_mc(dataset)[0]
        self.run_name = runs.get_run_name(dataset)
        self.run_number = runs.get_run_number(dataset)
        self.run_start = runs.get_run_start(dataset)
        loop_over_dataset(dataset, self.process_event,
                          event_lists=event_list,
                          branch_selection=self.branch_selection,
                          desc='Making %s minitree' % self.__class__.__name__)
        self.check_cache(force_empty=True)

        treemaker, already_made, minitree_path = hax.minitrees.check(
            self.run_name, self.EventLevelTreeMaker)
        if not len(self.data_event):
            log.warning(
                "Not a single event was extracted from dataset %s!" % dataset)
            return pd.DataFrame([], columns=['event_number', 'run_number'])
        elif (not already_made):  # and (event_list == None)
            hax.log.debug("Extraction completed, now saveing event data")
            # Manually call save_data
            metadata_dict = dict(
                version=self.EventLevelTreeMaker.__version__,
                extra=self.EventLevelTreeMaker.extra_metadata,
                pax_version=hax.paxroot.get_metadata(
                    self.run_name)['file_builder_version'],
                hax_version=hax.__version__,
                created_by=get_user_id(),
                event_list=event_list,
                documentation=self.EventLevelTreeMaker.__doc__,
                timestamp=str(datetime.now()))
            global metadata_dict

            get_format(minitree_path, self.EventLevelTreeMaker). \
                save_data(metadata_dict, pd.concat(
                    self.data_event, ignore_index=True))

        if not len(self.data):
            log.warning(
                "Not a single peak was extracted from dataset %s!" % dataset)
            return pd.DataFrame([], columns=['event_number', 'run_number'])
        else:
            hax.log.debug("Extraction completed, now concatenating peak data")
            return pd.concat(self.data, ignore_index=True)
