import scipy
from scipy import signal
import os
import lmdb
import pickle
import numpy as np
import mne

import pdb
from collections import defaultdict

# --- 통계용 카운터 ---
total_edf_processed = 0                 # 전체 처리한 EDF 수
task_edf_counts = defaultdict(int)      # task별 처리한 EDF 수
task_bz_list = defaultdict(list)        # task별 bz(=data.shape[0], 필터 전) 기록
task_kept_list = defaultdict(list)      # task별 kept(=event != 1로 남은 샘플 수) 기록
task_saved_samples = defaultdict(int)   # task별 최종 저장 샘플 수(누적)

mne.set_log_level('WARNING')

def load_xyz_from_elc(elc_path: str,
                      want_channels: list[str]) -> np.ndarray:
    want_up = [ch.upper() for ch in want_channels]

    # --- 앞 4줄 skip 후 파일 읽기 --------------------------
    with open(elc_path, 'r') as f:
        lines = f.readlines()[4:]

    positions_start = labels_start = None
    for i, ln in enumerate(lines):
        ll = ln.strip().lower()
        if ll == "positions":
            positions_start = i + 1
        elif ll == "labels":
            labels_start = i + 1
            break
    if positions_start is None or labels_start is None:
        raise RuntimeError("ELC 파일에 Positions/Labels 섹션이 없습니다.")

    # --- 좌표 읽기 -----------------------------------------
    positions = []
    for ln in lines[positions_start:labels_start-1]:
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        xyz = [float(p) for p in ln.split()[:3]]
        positions.append(np.array(xyz, dtype=float))

    # --- 라벨 읽기 -----------------------------------------
    labels = [ln.strip().upper() for ln in lines[labels_start:]
              if ln.strip() and not ln.startswith('#')]

    if len(labels) != len(positions):
        raise RuntimeError("Labels 개수와 Positions 개수가 다릅니다.")

    # --- 매핑 (want 순서 유지) ------------------------------
    xyz_list = []
    for ch in want_up:
        if ch in labels:
            idx = labels.index(ch)
            xyz_list.append(positions[idx])
        else:
            print(f"[ELC] Warning: {ch} not found; NaN inserted")
            xyz_list.append(np.full(3, np.nan))
    return np.vstack(xyz_list)          # shape = (len(want), 3)

tasks = ['04', '06', '08', '10', '12', '14']

root_dir = '/global/cfs/cdirs/m4750/DIVER/DOWNLOAD_DATASETS_MOVE_TO_M4750_LATER/PhysioNet/physionet.org/files/eegmmidb/1.0.0'

# ✅ split 제거: 전체 subject 디렉토리만 정렬해서 한 번에 처리
files = sorted([f for f in os.listdir(root_dir)])

# ✅ keys도 split 없이 하나로만 저장
dataset = []

selected_channels = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
                     'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
                     'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
                     'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
                     'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
                     'O1..', 'Oz..', 'O2..', 'Iz..']

db = lmdb.open('/pscratch/sd/t/tylee/Dataset/PhysioNet_200Hz_lowpass40_for_SOLID', map_size=12614542346)

# ELC 파일 경로
elc_file = "/pscratch/sd/t/tylee/standard_1005.elc"

# 원하는 채널 리스트 (저장용은 대문자로 만들 예정)
want_channels = ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5', 'C3', 'C1', 'Cz', 'C2',
                 'C4', 'C6', 'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6', 'Fp1', 'Fpz', 'Fp2',
                 'Af7', 'Af3', 'Afz', 'Af4', 'Af8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4',
                 'F6', 'F8', 'Ft7', 'Ft8', 'T7', 'T8', 'T9', 'T10', 'Tp7', 'Tp8', 'P7', 'P5',
                 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8',
                 'O1', 'Oz', 'O2', 'Iz']

# ✅ 저장되는 채널명은 전부 대문자
want_channels_up = [ch.upper() for ch in want_channels]

# xyz 로딩 (함수 내부에서 매칭은 대문자로 처리함)
xyz_array = load_xyz_from_elc(elc_file, want_channels)

for file in files:
    file_dir = os.path.join(root_dir, file)

    # 디렉토리가 아닌 경우 스킵
    if not os.path.isdir(file_dir):
        print(f"스킵됨 (디렉토리 아님): {file_dir}")
        continue

    for task in tasks:
        edf_path = os.path.join(file_dir, f'{file}R{task}.edf')

        if not os.path.exists(edf_path):
            print(f"EDF 파일 없음: {edf_path}")
            continue

        raw = mne.io.read_raw_edf(edf_path, preload=True)
        raw.pick_channels(selected_channels, ordered=True)

        if len(raw.info['bads']) > 0:
            print('interpolate_bads')
            raw.interpolate_bads()

        original_sampling_rate = int(raw.info['sfreq'])

        raw.set_eeg_reference(ref_channels='average')
        raw.filter(l_freq=0.3, h_freq=40)
        raw.notch_filter((60))
        raw.resample(200)  # set resample rate

        events_from_annot, event_dict = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw,
                            events_from_annot,
                            event_dict,
                            tmin=0,
                            tmax=4. - 1.0 / raw.info['sfreq'],
                            baseline=None,
                            preload=True)

        data = epochs.get_data(units='uV')
        events = epochs.events[:, 2]
        print(data.shape, events)

        data = data[:, :, -800:]  # -4*resample
        print(data.shape)

        bz, ch_nums, _ = data.shape
        data = data.reshape(bz, ch_nums, 4, 200)  # last should be resample rate

        # 통계 업데이트 (해당 EDF 하나에 대한 값)
        total_edf_processed += 1
        task_edf_counts[task] += 1
        task_bz_list[task].append(bz)

        kept = int(np.sum(events != 1))
        task_kept_list[task].append(kept)

        print(data.shape)

        for i, (sample, event) in enumerate(zip(data, events)):
            if event != 1:
                sample_key = f'{file}R{task}-{i}'

                data_dict = {
                    'sample': sample,
                    'label': event - 2 if task in ['04', '08', '12'] else event,
                    "data_info": {
                        "Dataset": "PhysioNet-MI",
                        "modality": "EEG",
                        "release": None,
                        "subject_id": file,
                        "task": 'PhysioNet-MI',
                        "resampling_rate": 200,
                        "original_sampling_rate": original_sampling_rate,
                        "segment_index": i,
                        "start_time": i * 4,
                        # ✅ 저장되는 채널 이름 리스트: 전부 대문자
                        "channel_names": want_channels_up,
                        "xyz_id": xyz_array
                    }
                }

                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()

                task_saved_samples[task] += 1
                # ✅ split 없이 그냥 전부 여기에 저장
                dataset.append(sample_key)

# ✅ keys도 split 없이 단일 리스트로 저장
txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()

print("\n" + "="*60)
print("[처리 통계 요약]")

print(f"- 총 처리한 EDF 개수: {total_edf_processed}")

print("- task별 처리한 EDF 개수:")
for t in tasks:
    print(f"  * R{t}: {task_edf_counts[t]}")

print("- task별 bz(필터 전 에포크 수) 요약:")
for t in tasks:
    arr = np.array(task_bz_list[t], dtype=int) if len(task_bz_list[t]) > 0 else np.array([])
    if arr.size > 0:
        print(f"  * R{t}: count={arr.size}, mean={arr.mean():.2f}, min={arr.min()}, max={arr.max()}")
    else:
        print(f"  * R{t}: (처리 없음)")

print("- task별 kept(event!=1 후 남은 샘플 수) 요약:")
for t in tasks:
    arr = np.array(task_kept_list[t], dtype=int) if len(task_kept_list[t]) > 0 else np.array([])
    if arr.size > 0:
        print(f"  * R{t}: count={arr.size}, mean={arr.mean():.2f}, min={arr.min()}, max={arr.max()}")
    else:
        print(f"  * R{t}: (처리 없음)")

print("- task별 최종 저장된 샘플(누적) 수:")
for t in tasks:
    print(f"  * R{t}: {task_saved_samples[t]}")

print("="*60 + "\n")

db.close()
