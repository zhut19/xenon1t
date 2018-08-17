from datetime import datetime

import hax
from hax import runs
from hax.minitrees import TreeMaker
from hax.corrections_handler import CorrectionsHandler
from hax.paxroot import loop_over_dataset
from hax.utils import find_file_in_folders, get_user_id
from hax.minitree_formats import get_format

import pickle

import numpy as np
from numpy import sqrt, exp, pi, square
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

        # BUG have incompatibility issure with dask
        self.cache = dict(peak=[], event=[])
        self.data = dict(peak=[], event=[])

    # Logistic part
    def process_event(self, event):
        result_peaks, result_event = self.extract_data(event)
        if not isinstance(result_peaks, (list, tuple)):
            raise TypeError("MultipleRowExtractor treemakers must extract "
                            "a list of dictionaries, not a %s" % type(result))
        assert len(result_peaks) == 0 or isinstance(result_peaks[0], dict)
        self.cache['peak'].extend(result_peaks)
        self.check_cache(level='peak')

        if not isinstance(result_event, dict):
            raise ValueError("TreeMakers must always extract dictionary")
        self.cache['event'].append(result_event)
        self.check_cache(level='event')

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

        self.check_cache(level='event', force_empty=True)
        treemaker, already_made, minitree_path = hax.minitrees.check(
            self.run_name, self.EventLevelTreeMaker)
        if not len(self.data['event']):
            log.warning(
                "Not a single row was extracted from dataset %s!" % dataset)
        elif (not already_made) and (event_list == None):
            metadata_dict = dict(
                version=self.EventLevelTreeMaker.__version__,
                extra=self.EventLevelTreeMaker.extra_metadata,
                pax_version=hax.paxroot.get_metadata(
                    self.run_name)['file_builder_version'],
                hax_version=hax.__version__,
                created_by=get_user_id(),
                event_list=event_list,
                documentation=self.EventLevelTreeMaker.__doc__,
                timestamp=str(
                    datetime.now()))

            get_format(minitree_path, self.EventLevelTreeMaker). \
                save_data(metadata_dict, pd.concat(
                    self.data['event'], ignore_index=True))

        self.check_cache(level='peak', force_empty=True)
        if not len(self.data['peak']):
            log.warning(
                "Not a single row was extracted from dataset %s!" % dataset)
            return pd.DataFrame([], columns=['event_number', 'run_number'])
        else:
            hax.log.debug("Extraction completed, now concatenating data")
            return pd.concat(self.data['peak'], ignore_index=True)

    def check_cache(self, level, force_empty=False):
        if not len(self.cache[level]) or (len(self.cache[level]) < self.cache_size and not force_empty):
            return
        self.data[level].append(pd.DataFrame(self.cache[level]))
        self.cache[level] = []


class MultipleS2Corrections(TreeMaker):
    """
    Provides:
     - s2_multi_peak
     - {cs2\cs2t\cs2b}_multi_peak
     - cs1_multi_peak
     - {x\y}_{tpf\nn}_multi_peak (for approximate FV selection)
    """
    __version__ = '6.0.1'


class MultipleS2Peaks(TwoLevelTreeMaker):
    __version__ = '6.0.0'
    extra_branches = ['peaks.*']

    sample_duration = 10  # ns
    drift_time_gate = 1.7e3  # ns
    drift_velocity_liquid = 0.0001335  # cm / ns

    with open('/project2/lgrandi/zhut/s2_single_classifier_gmix_v6.10.0.pkl', 'rb') as f:
        gmix_pattern = pickle.load(f)
    with open('/project2/lgrandi/zhut/s2_width_classifier_gmix_v6.10.0.pkl', 'rb') as f:
        gmix_width = pickle.load(f)

    corrections_handler = CorrectionsHandler()

    EventLevelTreeMaker = MultipleS2Corrections

    def s2_width_model(z):
        z[z > 0] = -1e3  # Hope this can conter

        w0 = 229.58  # 309.7/1.349
        coeff = 0.925
        dif_const = 31.73
        v = .1335
        return sqrt(square(w0) - 2 * coeff * dif_const * z / v ** 3) * 1.349

    def determine_interaction(self, df):
        df['not_interaction_depth'] = (df['z'] > 0) | (df['z'] < -100)

        X = np.log10(df.loc[:, ['area', 'goodness_of_fit_tpf', 's2']])
        df['not_interaction_pattern'] = np.array(
            MultipleS2Peaks.gmix_pattern.predict(X), bool)

        X = np.column_stack([np.log10(
            df['area']), df['range_50p_area']/MultipleS2Peaks.s2_width_model(df['z'])])
        df['not_interaction_width'] = np.array(
            MultipleS2Peaks.gmix_width.predict(X), bool)

        df['not_interaction'] = np.logical_or(
            df.not_interaction_pattern,  df.not_interaction_width)
        df['not_interaction'] = np.logical_or(
            df.not_interaction,  df.not_interaction_depth)
        return df

    def correction(self, df, run_number, run_start, mc_data):
        """
        Here load correction maps 
        Use both tpf and nn position to do corrections for S2 peaks

        Provides:
         - s2_lifetime_correction
         - s2_xy_{tpf\nn}_correction_{tot\top\bottom}
         - s1_xyz_correction_{tpf_fdc_2d\nn_fdc_3d}
         - carea_{tpf\nn}
        """

        df['s2_lifetime_correction'] = (
            MultipleS2Peaks.corrections_handler.get_electron_lifetime_correction(
                run_number, run_start, df.drift_time, mc_data))

        s2_xy_tpf_correction_tot = np.ones(len(df))
        s2_xy_tpf_correction_top = np.ones(len(df))
        s2_xy_tpf_correction_bottom = np.ones(len(df))
        s2_xy_nn_correction_tot = np.ones(len(df))
        s2_xy_nn_correction_top = np.ones(len(df))
        s2_xy_nn_correction_bottom = np.ones(len(df))
        s1_xyz_correction_tpf_fdc_2d = np.ones(len(df))
        s1_xyz_correction_nn_fdc_3d = np.ones(len(df))

        p_i = 0
        for i, p in df.iterrows():
            s2_xy_tpf_correction_tot[p_i] = (
                MultipleS2Peaks.corrections_handler.get_correction_from_map(
                    "s2_xy_map", run_number, [p.x_tpf, p.y_tpf]))

            s2_xy_tpf_correction_top[p_i] = (
                MultipleS2Peaks.corrections_handler.get_correction_from_map(
                    "s2_xy_map", run_number, [p.x_tpf, p.y_tpf], map_name='map_top'))

            s2_xy_tpf_correction_bottom[p_i] = (
                MultipleS2Peaks.corrections_handler.get_correction_from_map(
                    "s2_xy_map", run_number, [p.x_tpf, p.y_tpf], map_name='map_bottom'))

            s2_xy_nn_correction_tot[p_i] = (
                MultipleS2Peaks.corrections_handler.get_correction_from_map(
                    "s2_xy_map", run_number, [p.x_nn, p.y_nn]))

            s2_xy_nn_correction_top[p_i] = (
                MultipleS2Peaks.corrections_handler.get_correction_from_map(
                    "s2_xy_map", run_number, [p.x_nn, p.y_nn], map_name='map_top'))

            s2_xy_nn_correction_bottom[p_i] = (
                MultipleS2Peaks.corrections_handler.get_correction_from_map(
                    "s2_xy_map", run_number, [p.x_nn, p.y_nn], map_name='map_top'))

            s1_xyz_correction_tpf_fdc_2d[p_i] = (
                1 / MultipleS2Peaks.corrections_handler.get_correction_from_map(
                    "s1_lce_map_tpf_fdc_2d", run_number, [p.x_tpf, p.y_tpf, p.z]))

            s1_xyz_correction_nn_fdc_3d[p_i] = (
                1 / MultipleS2Peaks.corrections_handler.get_correction_from_map(
                    "s1_lce_map_nn_fdc_3d", run_number, [p.x_nn, p.y_nn, p.z]))

            p_i += 1

        df['s2_xy_tpf_correction_tot'] = s2_xy_nn_correction_tot
        df['s2_xy_nn_correction_tot'] = s2_xy_nn_correction_tot

        df['s1_xyz_correction_tpf_fdc_2d'] = s1_xyz_correction_tpf_fdc_2d
        df['s1_xyz_correction_nn_fdc_3d'] = s1_xyz_correction_nn_fdc_3d

        df['carea_tpf'] = df.s2_lifetime_correction * \
            df.s2_xy_nn_correction_tot*df.area
        df['carea_nn'] = df.s2_lifetime_correction * \
            df.s2_xy_nn_correction_tot*df.area

        return df

    def summarize_to_event(self, df, event_data):
        """
        Now we can start another event minitree generation
        Which can be loaded minitrees

        Provides:
         - s2_multi_peak
         - {cs2\cs2t\cs2b}_multi_peak
         - cs1_multi_peak
         - {x\y}_{tpf\nn}_multi_peak (for approximate FV selection)
        """
        mask = np.logical_not(df.not_interaction)

        if len(df.loc[mask]) < 1:
            return event_data

        df.loc[mask, 'weighted_s1_xyz_correction_tpf_fdc_2d'] = (
            np.average(df.loc[mask, 's1_xyz_correction_tpf_fdc_2d'],
                       weights=df.loc[mask, 'carea_tpf']))
        df.loc[mask, 'weighted_s1_xyz_correction_nn_fdc_3d'] = (
            np.average(df.loc[mask, 's1_xyz_correction_nn_fdc_3d'],
                       weights=df.loc[mask, 'carea_tpf']))

        event_data['s2_multi_peak'] = np.sum(df.loc[mask, 'area'])
        event_data['cs2_multi_peak'] = np.sum(df.loc[mask, 'carea_nn'])
        event_data['cs2t_multi_peak'] = (
            np.sum(df.loc[mask, 'carea_nn'] * df.loc[mask, 'area_fraction_top']))
        event_data['cs2b_multi_peak'] = (
            np.sum(df.loc[mask, 'carea_nn'] * (1 - df.loc[mask, 'area_fraction_top'])))
        event_data['cs1_multi_peak'] = (
            np.average(df.loc[mask, 's1'] * df.loc[mask, 'weighted_s1_xyz_correction_nn_fdc_3d']))

        event_data['x_tpf_multi_peak'] = (
            np.average(df.loc[mask, 'x_tpf'], weights=df.loc[mask, 'carea_tpf']))
        event_data['y_tpf_multi_peak'] = (
            np.average(df.loc[mask, 'y_tpf'], weights=df.loc[mask, 'carea_tpf']))
        event_data['x_nn_multi_peak'] = (
            np.average(df.loc[mask, 'x_nn'], weights=df.loc[mask, 'carea_tpf']))
        event_data['y_nn_multi_peak'] = (
            np.average(df.loc[mask, 'y_nn'], weights=df.loc[mask, 'carea_tpf']))

        return event_data

    def extract_data(self, event):
        """
        At peak level this take after multirowextractor
        Extract S2 peaks in tpc that area>150 and happen after S1 of interaction[0]

        Provides:
         direct_fields (check below)
         not_direct_fields (check below)
         corrected_fields (check correction function above)
         s1, s2 of interaction[0]
        """

        event_data = dict(
            run_number=self.run_number,
            event_number=event.event_number)

        if not len(event.interactions):
            return [], event_data

        interaction = event.interactions[0]
        s1 = event.peaks[interaction.s1]
        s2 = event.peaks[interaction.s2]

        direct_fields = ['area', 'area_fraction_top',
                         'center_time', 'detector', 'index_of_maximum', 'type']

        not_direct_fields = ['goodness_of_fit_nn', 'goodness_of_fit_tpf',
                             'range_50p_area', 'x_nn', 'x_tpf', 'y_nn', 'y_tpf']

        number_peaks = len(event.peaks)
        peak_data = {field: list(np.zeros(number_peaks))
                     for field in direct_fields+not_direct_fields}

        for peak_i, peak in enumerate(event.peaks):

            peak_data['range_50p_area'][peak_i] = list(
                peak.range_area_decile)[5]

            for rp in peak.reconstructed_positions:
                if rp.algorithm == 'PosRecTopPatternFit':
                    for field in ['x', 'y', 'goodness_of_fit']:
                        peak_data[field+'_tpf'][peak_i] = getattr(rp, field)

                if rp.algorithm == 'PosRecNeuralNet':
                    for field in ['x', 'y', 'goodness_of_fit']:
                        peak_data[field+'_nn'][peak_i] = getattr(rp, field)

            for field in direct_fields:
                peak_data[field][peak_i] = getattr(peak, field)

        peaks = pd.DataFrame(peak_data)
        peaks['event_number'] = event.event_number
        peaks['run_number'] = self.run_number
        peaks['drift_time'] = (
            peaks.index_of_maximum - s1.index_of_maximum) * MultipleS2Peaks.sample_duration
        peaks['z'] = - MultipleS2Peaks.drift_velocity_liquid * \
            (peaks['drift_time'] - MultipleS2Peaks.drift_time_gate)
        peaks['s2'] = s2.area
        peaks['s1'] = s1.area

        peaks = peaks[peaks.eval(
            '(type=="s2") & (detector == "tpc") & (area>150) & (z<0)')]
        if len(peaks) < 1:
            return [], event_data

        peaks = self.correction(peaks, self.run_number,
                                self.run_start, self.mc_data)

        peaks = self.determine_interaction(peaks)

        event_data = self.summarize_to_event(peaks, event_data)

        return peaks.to_dict('records'), event_data
