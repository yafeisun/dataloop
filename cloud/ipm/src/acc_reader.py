import csv
import os.path
import numpy as np
from src.utils import log

class CanCsvReader:
    def __init__(self, can_path):
        ts_acc = {}
        if os.path.exists(can_path):
            with open(can_path) as f:
                reader = csv.reader(f)
                for r in reader:
                    if r[1] == 'ADataRawSafeALat':
                        ts = int(float(r[0]) * 1E9)
                        if ts not in ts_acc:
                            ts_acc[ts] = [ts, 0, 0]
                        ts_acc[ts][1] = float(r[2])
                    if r[1] == 'ADataRawSafeALgt':
                        ts = int(float(r[0]) * 1E9)
                        if ts not in ts_acc:
                            ts_acc[ts] = [ts, 0, 0]
                        ts_acc[ts][2] = float(r[2])
        if not ts_acc:
            log.warning(f'not found ADataRawSafeALat or  ADataRawSafeALgt in {can_path}')
        self.acc_data = sorted([v for v in ts_acc.values()], key=lambda x: x[0]) # ts, acc_alt, acc_lat
        self.acc_data = np.array(self.acc_data)

    def get_acc(self, ts_ns):
        if not len(self.acc_data):
            return None, (None, None)
        ix = np.searchsorted(self.acc_data[:, 0], ts_ns, side='left')
        if ix == 0:
            return self.acc_data[ix, 0], self.acc_data[ix, 1:]
        if ix == len(self.acc_data):
            return self.acc_data[-1, 0], self.acc_data[-1, 1:]

        pre_ix = ix - 1
        pre_delta_ts = ts_ns - self.acc_data[pre_ix, 0]
        next_delta_ts = self.acc_data[ix, 0] - ts_ns
        if pre_delta_ts < next_delta_ts:
            acc = self.acc_data[pre_ix, 1:]
            log.info(f'{acc} ts_image - ts_acc = {pre_delta_ts/1E6:.3f} ms')
            return self.acc_data[pre_ix, 0]/1E9, acc
        else:
            acc = self.acc_data[ix, 1:]
            log.info(f'{acc} ts_acc- ts_image = {next_delta_ts/1E6:.3f} ms')
            return self.acc_data[ix, 0]/1E9, acc
