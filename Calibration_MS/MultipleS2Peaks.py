from TwoLevel import TwoLevelTreeMaker
from hax.minitrees import TreeMaker
from hax.corrections_handler import CorrectionsHandler
import pickle

import numpy as np
import pandas as pd
from numpy import sqrt, square


class MultipleS2Corrections(TreeMaker):
    """
    Provides:
     - s2_multi_peak
     - {cs2\cs2t\cs2b}_multi_peak
     - cs1_multi_peak
     - {x\y}_{tpf\nn}_multi_peak (for approximate FV selection)
    """
    __version__ = '1.2.0'


class MultipleS2Peaks(TwoLevelTreeMaker):
    __version__ = '1.2.1'
    extra_branches = ['peaks.*']
    EventLevelTreeMaker = MultipleS2Corrections

    # Inputs for peak level info
    sample_duration = 10  # ns
    drift_time_gate = 1.7e3  # ns
    drift_velocity_liquid = 0.0001335  # cm / ns

    def s2_width_model(self, z):
        w0, dif_const, v = 229.58, 29.35, .1335
        return sqrt(square(w0) - 2 * dif_const * z / v ** 3) * 1.349

    # Inputs for correction
    corrections_handler = CorrectionsHandler()

    def determine_interaction(self, df):

        # Inputs for interaction s2 peaks selection
        try:
            self.gmix_pattern
        except AttributeError:
            with open('/project2/lgrandi/zhut/s2_single_classifier_gmix_v6.10.0.pkl', 'rb') as f:
                self.gmix_pattern = pickle.load(f)
        try:
            self.gmix_width
        except AttributeError:
            with open('/project2/lgrandi/zhut/s2_width_classifier_gmix_v6.10.0.pkl', 'rb') as f:
                self.gmix_width = pickle.load(f)

        df['not_interaction_depth'] = (df['z'] > 0) | (df['z'] < -100)

        df['not_interaction_pattern'] = False
        mask = df.goodness_of_fit_tpf > 0
        if len(df.loc[mask]) > 0:
            X = np.log10(df.loc[mask, ['area', 'goodness_of_fit_tpf', 's2']])
            df.loc[mask, 'not_interaction_pattern'] = np.array(
                self.gmix_pattern.predict(X), bool)

        df['not_interaction_width'] = False
        mask = (df.range_50p_area > 0) & (df.z < 0)
        if len(df.loc[mask]) > 0:
            X = np.column_stack([np.log10(
                df.loc[mask, 'area']), df.loc[mask, 'range_50p_area']
                / self.s2_width_model(df.loc[mask, 'z'])])
            df.loc[mask, 'not_interaction_width'] = np.array(
                self.gmix_width.predict(X), bool)

        df['not_interaction'] = df.not_interaction_pattern | df.not_interaction_width | df.not_interaction_depth
        return df

    def correction(self, df, run_number, run_start, mc_data):
        """
        Here load correction maps 
        Use both tpf and nn position to do corrections for S2 peaks

        Provides:
         - s2_lifetime_correction
         - s2_xy_{tpf\nn}_correction_{tot}
         - s1_xyz_correction_{tpf_fdc_2d\nn_fdc_3d}
         - carea_{tpf\nn}
        """

        df['s2_lifetime_correction'] = (
            self.corrections_handler.get_electron_lifetime_correction(
                run_number, run_start, df.drift_time, mc_data))

        s2_xy_tpf_correction_tot = np.ones(len(df))
        s2_xy_nn_correction_tot = np.ones(len(df))
        s1_xyz_correction_tpf_fdc_2d = np.ones(len(df))
        s1_xyz_correction_nn_fdc_3d = np.ones(len(df))

        p_i = 0
        for i, p in df.iterrows():
            s2_xy_tpf_correction_tot[p_i] = (
                self.corrections_handler.get_correction_from_map(
                    "s2_xy_map", run_number, [p.x_tpf, p.y_tpf]))

            s2_xy_nn_correction_tot[p_i] = (
                self.corrections_handler.get_correction_from_map(
                    "s2_xy_map", run_number, [p.x_nn, p.y_nn]))

            s1_xyz_correction_tpf_fdc_2d[p_i] = (
                1 / self.corrections_handler.get_correction_from_map(
                    "s1_lce_map_tpf_fdc_2d", run_number, [p.x_tpf, p.y_tpf, p.z]))

            s1_xyz_correction_nn_fdc_3d[p_i] = (
                1 / self.corrections_handler.get_correction_from_map(
                    "s1_lce_map_nn_fdc_3d", run_number, [p.x_nn, p.y_nn, p.z]))

            p_i += 1

        df['s2_xy_tpf_correction_tot'] = s2_xy_tpf_correction_tot
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
        Now we can start another minitree generation
        Which can fit into standard minitree

        Provides:
         - s2_multi_peak
         - {cs2\cs2t\cs2b}_multi_peak
         - cs1_multi_peak
         - {x\y}_{tpf\nn}_multi_peak (for approximate FV selection)
         - n_multi_peak
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
        event_data['n_multi_peak'] = len(df.loc[mask])

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
            peaks.index_of_maximum - s1.index_of_maximum) * self.sample_duration
        peaks['z'] = -self.drift_velocity_liquid * \
            (peaks['drift_time'] - self.drift_time_gate)
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
