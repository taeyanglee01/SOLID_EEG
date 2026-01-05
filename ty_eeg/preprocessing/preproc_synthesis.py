import os
import lmdb
import pickle
import numpy as np
import mne

from collections import defaultdict

mne.set_log_level('WARNING')

# =========================
# 통계용 카운터
# =========================
total_fif_processed = 0
task_fif_counts = defaultdict(int)
task_bz_list = defaultdict(list)        # bz(=epoch/window 개수)
task_kept_list = defaultdict(list)      # kept(=label 조건 통과 수)
task_saved_samples = defaultdict(int)   # 최종 저장 샘플 누적

# =========================
# ELC -> XYZ
# =========================
def load_xyz_from_elc(elc_path: str, want_channels: list[str]) -> np.ndarray:
    want_up = [ch.upper() for ch in want_channels]
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

    positions = []
    for ln in lines[positions_start:labels_start-1]:
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        xyz = [float(p) for p in ln.split()[:3]]
        positions.append(np.array(xyz, dtype=float))

    labels = [ln.strip().upper() for ln in lines[labels_start:]
              if ln.strip() and not ln.startswith('#')]

    if len(labels) != len(positions):
        raise RuntimeError("Labels 개수와 Positions 개수가 다릅니다.")

    xyz_list = []
    for ch in want_up:
        if ch in labels:
            idx = labels.index(ch)
            xyz_list.append(positions[idx])
        else:
            print(f"[ELC] Warning: {ch} not found; NaN inserted")
            xyz_list.append(np.full(3, np.nan))
    return np.vstack(xyz_list)  # (len(want), 3)

# =========================
# events.txt 로드 (유연하게)
# =========================
def load_events_txt(events_path: str, sfreq: float):
    """
    events.txt 지원 포맷:
      - onset_sec label
      - onset_sec duration_sec label
      - onset_sample label  (onset 값이 크면 sample로 간주)
    반환: mne events (n,3), label_raw_values
    """
    arr = np.loadtxt(events_path)
    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.shape[1] == 2:
        onset = arr[:, 0]
        label = arr[:, 1].astype(int)
    elif arr.shape[1] == 3:
        onset = arr[:, 0]
        label = arr[:, 2].astype(int)
    else:
        raise ValueError(f"Unsupported events.txt columns: {arr.shape[1]} in {events_path}")

    # onset이 sample인지 sec인지 추정
    if np.nanmax(onset) > 1e5:
        onset_samp = onset.astype(int)
    else:
        onset_samp = np.round(onset * sfreq).astype(int)

    events = np.column_stack([onset_samp,
                              np.zeros(len(onset_samp), dtype=int),
                              label]).astype(int)
    return events, label

# =========================
# 연속신호 window 자르기 (events 없을 때)
# =========================
def make_windows(raw, win_sec=4.0, stride_sec=4.0):
    sfreq = raw.info['sfreq']
    win = int(round(win_sec * sfreq))
    stride = int(round(stride_sec * sfreq))
    data = raw.get_data(units='uV')  # (ch, time)
    ch, T = data.shape
    starts = np.arange(0, max(T - win + 1, 0), stride, dtype=int)
    if len(starts) == 0:
        return np.zeros((0, ch, win), dtype=np.float32), np.zeros((0,), dtype=int)

    X = np.stack([data[:, s:s+win] for s in starts], axis=0).astype(np.float32)  # (bz, ch, time)
    y = np.full((X.shape[0],), -1, dtype=int)
    return X, y

# =========================
# 메인 설정
# =========================
root_dir = "/pscratch/sd/t/tylee/SOLID_EEG_RESULT/synthetic_eeg/multisubject_data"

# sub-001 ~ sub-050
subjects = [f"sub-{i:03d}" for i in range(1, 51)]

# 네 synthetic 폴더에 있는 task prefix들
tasks = [
    # "emotion_negative",
    # "emotion_neutral",
    # "emotion_positive",
    #"motor_imagery_both_hands",
    #"motor_imagery_feet",
    #"motor_imagery_left_hand",
    #"motor_imagery_right_hand",
    #"p300_oddball",
    "seizure_absence",
    "seizure_focal_frontal",
    "seizure_focal_temporal",
]

# groundtruth 쓸지 여부 (원하면 True로)
use_groundtruth = False  # True면 *_groundtruth_raw.fif 우선 사용

# 전처리 파라미터
l_freq = 0.3
h_freq = 40
notch = 60.0
resample_rate = 200

# Epoch/window 길이
epoch_sec = 4.0

# 저장 DB
db = lmdb.open("/pscratch/sd/t/tylee/Dataset/SYNTHETIC_Seizure_EEG_200Hz_lowpass40_for_SOLID",
               map_size=12614542346)

# keys는 split 없이 단일 리스트
dataset = []

# 채널/xyz
elc_file = "/pscratch/sd/t/tylee/standard_1005.elc"
want_channels = ['Fp1', 'Fpz', 'Fp2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFz', 'AF2', 'AF4', 'AF6',
 'AF8', 'AF10', 'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F10', 'FT9', 'FT7', 'FC5',
 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
 'C6', 'T8', 'T10', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P9',
 'P7', 'P5', 'P3', 'P1', 'Pz']
want_channels_up = [ch.upper() for ch in want_channels]
xyz_array = load_xyz_from_elc(elc_file, want_channels)

# =========================
# Loop
# =========================
for sub in subjects:
    sub_dir = os.path.join(root_dir, sub)

    if not os.path.isdir(sub_dir):
        print(f"스킵됨 (디렉토리 아님): {sub_dir}")
        continue

    for task in tasks:
        # 입력 fif 선택
        fif_gt = os.path.join(sub_dir, f"{task}_groundtruth_raw.fif")
        fif_raw = os.path.join(sub_dir, f"{task}_raw.fif")

        if use_groundtruth and os.path.exists(fif_gt):
            fif_path = fif_gt
        elif os.path.exists(fif_raw):
            fif_path = fif_raw
        elif os.path.exists(fif_gt):
            fif_path = fif_gt
        else:
            # 해당 task 없음
            continue

        # events.txt 있으면 사용
        events_path = os.path.join(sub_dir, f"{task}_events.txt")
        # has_events_txt = os.path.exists(events_path)
        has_events_txt = False # FIXME : epoching is not complete yet

        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)

        # (선택) 채널 픽: want_channels(대소문자 혼재 가능) 기준으로 존재하는 채널만 유지
        # synthetic 채널명이 이미 표준이면 그대로 두는 편이 안전.
        # 아래는 존재하는 채널만 골라서 want 순서로 정렬 시도.
        want_in_raw = []
        raw_ch_upper = {c.upper(): c for c in raw.info['ch_names']}
        for ch in want_channels:
            if ch.upper() in raw_ch_upper:
                want_in_raw.append(raw_ch_upper[ch.upper()])
        if len(want_in_raw) > 0:
            raw.pick_channels(want_in_raw, ordered=True)

        if len(raw.info.get('bads', [])) > 0:
            print("interpolate_bads")
            raw.interpolate_bads()

        original_sampling_rate = int(raw.info['sfreq'])

        # 전처리 (네 코드와 동일 흐름)
        raw.set_eeg_reference(ref_channels='average')
        raw.filter(l_freq=l_freq, h_freq=h_freq)
        raw.notch_filter((notch))
        raw.resample(resample_rate)

        # ====== Epoch 생성 ======
        if has_events_txt:
            # events.txt 기반
            events_from_txt, labels_raw = load_events_txt(events_path, raw.info['sfreq'])
            # (중요) MNE Epochs는 event_id dict가 필요하므로 label값 그대로 매핑
            uniq = sorted(np.unique(events_from_txt[:, 2]).tolist())
            event_id = {str(v): int(v) for v in uniq}

            epochs = mne.Epochs(
                raw,
                events_from_txt,
                event_id=event_id,
                tmin=0,
                tmax=epoch_sec - 1.0 / raw.info['sfreq'],
                baseline=None,
                preload=True,
                reject_by_annotation=False,
                verbose=False
            )
            data = epochs.get_data(units='uV')  # (bz, ch, time)
            events = epochs.events[:, 2].astype(int)

        else:
            # annotations 있으면 그걸로, 없으면 window로
            if raw.annotations is not None and len(raw.annotations) > 0:
                events_from_annot, event_dict = mne.events_from_annotations(raw)
                if len(events_from_annot) > 0:
                    epochs = mne.Epochs(
                        raw,
                        events_from_annot,
                        event_dict,
                        tmin=0,
                        tmax=epoch_sec - 1.0 / raw.info['sfreq'],
                        baseline=None,
                        preload=True,
                        reject_by_annotation=False,
                        verbose=False
                    )
                    data = epochs.get_data(units='uV')
                    events = epochs.events[:, 2].astype(int)
                else:
                    data, events = make_windows(raw, win_sec=epoch_sec, stride_sec=epoch_sec)
            else:
                data, events = make_windows(raw, win_sec=epoch_sec, stride_sec=epoch_sec)

        # ====== 네 코드와 동일 reshape 처리 ======
        # time축에서 마지막 4초만 취함 (resample_rate * 4)
        # 혹시 epoch이 이미 4초면 그냥 그대로이고, 더 길면 뒤를 자름
        need = int(resample_rate * epoch_sec)
        if data.shape[-1] >= need:
            data = data[:, :, -need:]
        else:
            # 너무 짧으면 skip
            print(f"[SKIP] too short: {sub} {task} data_len={data.shape[-1]} need={need}")
            continue

        bz, ch_nums, _ = data.shape
        data = data.reshape(bz, ch_nums, int(epoch_sec), resample_rate)

        # ====== 통계 업데이트(파일 1개 단위) ======
        total_fif_processed += 1
        task_fif_counts[task] += 1
        task_bz_list[task].append(bz)

        # kept 조건: 네 코드대로 event != 1
        kept = int(np.sum(events != 1))
        task_kept_list[task].append(kept)

        print(f"[{sub} | {task}] data={data.shape} kept={kept}")

        # ====== 저장 ======
        for i, (sample, event) in enumerate(zip(data, events)):
            if event != 1:
                sample_key = f"{sub}/{task}-{i}"

                data_dict = {
                    "sample": sample,  # (ch, 4, resample_rate)
                    "label": int(event),  # 너 로직 유지. 필요하면 task별로 재매핑 가능
                    "data_info": {
                        "Dataset": "Synthetic-EEG",
                        "modality": "EEG",
                        "release": None,
                        "subject_id": sub,
                        "task": task,
                        "source_file": os.path.basename(fif_path),
                        "resampling_rate": int(resample_rate),
                        "original_sampling_rate": original_sampling_rate,
                        "segment_index": i,
                        "start_time": i * epoch_sec,
                        "channel_names": want_channels_up,
                        "xyz_id": xyz_array,
                        "events_source": "events.txt" if has_events_txt else ("annotations" if (raw.annotations is not None and len(raw.annotations) > 0) else "window"),
                    }
                }

                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()

                task_saved_samples[task] += 1
                dataset.append(sample_key)

# ====== keys 저장 ======
txn = db.begin(write=True)
txn.put(key=b'__keys__', value=pickle.dumps(dataset))
txn.commit()

# ====== 통계 출력 ======
print("\n" + "="*60)
print("[처리 통계 요약]")

print(f"- 총 처리한 FIF 개수: {total_fif_processed}")

print("- task별 처리한 FIF 개수:")
for t in tasks:
    print(f"  * {t}: {task_fif_counts[t]}")

print("- task별 bz(에포크/윈도우 개수) 요약:")
for t in tasks:
    arr = np.array(task_bz_list[t], dtype=int) if len(task_bz_list[t]) > 0 else np.array([])
    if arr.size > 0:
        print(f"  * {t}: count={arr.size}, mean={arr.mean():.2f}, min={arr.min()}, max={arr.max()}")
    else:
        print(f"  * {t}: (처리 없음)")

print("- task별 kept(event!=1 후 남은 샘플 수) 요약:")
for t in tasks:
    arr = np.array(task_kept_list[t], dtype=int) if len(task_kept_list[t]) > 0 else np.array([])
    if arr.size > 0:
        print(f"  * {t}: count={arr.size}, mean={arr.mean():.2f}, min={arr.min()}, max={arr.max()}")
    else:
        print(f"  * {t}: (처리 없음)")

print("- task별 최종 저장된 샘플(누적) 수:")
for t in tasks:
    print(f"  * {t}: {task_saved_samples[t]}")

print("="*60 + "\n")

db.close()
