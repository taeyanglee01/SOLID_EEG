# Synthetic EEG Preprocessing and LMDB Conversion

이 가이드는 생성된 synthetic EEG 데이터를 전처리하고 LMDB 형식으로 변환하는 방법을 설명합니다.

## 워크플로우

```
1. Multi-subject 데이터 생성
   ↓
2. 전처리 + LMDB 변환
   ↓
3. LMDB 검증
   ↓
4. 모델 학습에 사용
```

## 파일 구조

```
synthetic/
├── generate_task_specific_eeg_multisubject.py  # 데이터 생성
├── preproc_synthetic_to_lmdb.py                # 전처리 + LMDB 변환
├── verify_lmdb.py                              # LMDB 검증
├── run_preproc_to_lmdb.sh                      # SLURM 스크립트
└── LMDB_PREPROCESSING_README.md                # 이 파일
```

## 1. 데이터 생성 (이미 완료한 경우 스킵)

```bash
python generate_task_specific_eeg_multisubject.py \
    --n_subjects 100 \
    --output_dir /path/to/multisubject_data
```

생성된 구조:
```
multisubject_data/
├── sub-001/
│   ├── motor_imagery_left_hand_raw.fif
│   ├── motor_imagery_left_hand_groundtruth_raw.fif
│   ├── motor_imagery_left_hand_events.txt
│   ├── ...
│   └── emotion_neutral_raw.fif
├── sub-002/
└── ...
```

## 2. 전처리 및 LMDB 변환

### 2.1 전처리 과정

각 데이터에 대해 다음 전처리가 적용됩니다:

1. **Bad channel interpolation** - 마킹된 bad channels 보간
2. **Common Average Reference (CAR)** - 평균 참조 전극
3. **High-pass filter** (0.3 Hz) - 선택적 (기본: 꺼짐)
4. **Resampling** (200 Hz) - 기존 250 Hz → 200 Hz
5. **Epoching** - Task별로 다른 방식
6. **Reshape** - (n_channels, 4, 200) 형태로 변환

### 2.2 Task별 처리 방식

#### Motor Imagery
- Event 기반 epoching (4초 구간)
- Label: 0=left_hand, 1=right_hand, 2=both_hands, 3=feet

#### Seizure
- Seizure onset 기반 epoching (10초 → 4초 segments로 분할)
- Label: 1 (seizure)

#### P300 Oddball
- Target stimulus 기반 epoching (1초 × 4 = 4초)
- Label: 1 (target)

#### Emotion
- Continuous 데이터를 4초 segments로 분할
- Label: 0=positive, 1=negative, 2=neutral

### 2.3 실행 방법

#### 옵션 1: 직접 실행

```bash
python preproc_synthetic_to_lmdb.py \
    --input_dir /path/to/multisubject_data \
    --output_lmdb /path/to/output.lmdb \
    --tasks all \
    --target_sfreq 200 \
    --map_size 50000000000
```

#### 옵션 2: SLURM 제출

```bash
# run_preproc_to_lmdb.sh 수정 (경로 설정)
sbatch run_preproc_to_lmdb.sh
```

### 2.4 Task 선택

특정 task만 처리하려면:

```bash
# Motor Imagery만
python preproc_synthetic_to_lmdb.py \
    --input_dir /path/to/data \
    --output_lmdb /path/to/motor_imagery.lmdb \
    --tasks motor_imagery

# 여러 task
python preproc_synthetic_to_lmdb.py \
    --input_dir /path/to/data \
    --output_lmdb /path/to/combined.lmdb \
    --tasks motor_imagery,p300

# 모든 task
python preproc_synthetic_to_lmdb.py \
    --input_dir /path/to/data \
    --output_lmdb /path/to/all_tasks.lmdb \
    --tasks all
```

## 3. LMDB 검증

생성된 LMDB가 올바른지 확인:

```bash
python verify_lmdb.py \
    --lmdb_path /path/to/output.lmdb \
    --n_samples 10
```

출력 예시:
```
================================================================================
LMDB Database Verification
================================================================================
Database path: /path/to/output.lmdb

Total samples in database: 15000

Key analysis:

Number of subjects: 100
Subjects: ['sub-001', 'sub-002', ..., 'sub-100']

Samples per task:
  motor_imagery_left_hand: 600
  motor_imagery_right_hand: 600
  ...

Sample Data Verification
================================================================================
Sample 1/10: sub-001_motor_imagery_left_hand-0
  Sample shape: (64, 4, 200)
  Sample dtype: float64
  Sample range: [-45.23, 52.11] µV
  Label: 0
  Dataset: Synthetic-MI
  Subject: sub-001
  Task: motor_imagery_left_hand
  Resampling rate: 200 Hz
  Number of channels: 64
  XYZ coordinates shape: (64, 3)

✓ All checks passed!
```

## 4. LMDB 데이터 구조

### Key 형식
```
{subject_id}_{task_name}-{segment_index}
```

예시:
- `sub-001_motor_imagery_left_hand-0`
- `sub-025_p300_oddball-42`
- `sub-100_emotion_positive-5`

### Value 구조 (Pickle 형식)

```python
{
    'sample': np.ndarray,  # shape: (n_channels, 4, 200)
                          # n_channels=64, 4초, 200Hz

    'label': int,         # Task별 label

    'data_info': {
        'Dataset': str,              # 'Synthetic-MI', 'Synthetic-Seizure', etc.
        'modality': 'EEG',
        'release': 'synthetic',
        'subject_id': str,           # 'sub-001'
        'task': str,                 # 'motor_imagery_left_hand'
        'resampling_rate': int,      # 200
        'original_sampling_rate': int,  # 250
        'segment_index': int,        # Epoch/segment number
        'start_time': float,         # Start time in seconds
        'channel_names': list,       # ['FC5', 'FC3', ..., 'IZ']
        'xyz_id': np.ndarray        # shape: (64, 3), 채널 좌표
    }
}
```

## 5. 모델에서 LMDB 사용하기

### 5.1 LMDB 로드 예제

```python
import lmdb
import pickle
import numpy as np

# Open database
db = lmdb.open('/path/to/database.lmdb', readonly=True, lock=False)
txn = db.begin()

# Load all keys
keys = pickle.loads(txn.get('__keys__'.encode()))
print(f"Total samples: {len(keys)}")

# Load a sample
sample_key = keys[0]
data_dict = pickle.loads(txn.get(sample_key.encode()))

sample = data_dict['sample']  # (64, 4, 200)
label = data_dict['label']
data_info = data_dict['data_info']

print(f"Sample shape: {sample.shape}")
print(f"Label: {label}")
print(f"Subject: {data_info['subject_id']}")
print(f"Task: {data_info['task']}")
```

### 5.2 PyTorch DataLoader 예제

```python
import torch
from torch.utils.data import Dataset, DataLoader
import lmdb
import pickle

class SyntheticEEGDataset(Dataset):
    def __init__(self, lmdb_path):
        self.db = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.txn = self.db.begin()
        self.keys = pickle.loads(self.txn.get('__keys__'.encode()))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data_dict = pickle.loads(self.txn.get(key.encode()))

        sample = torch.FloatTensor(data_dict['sample'])
        label = torch.LongTensor([data_dict['label']])

        return sample, label

# Create DataLoader
dataset = SyntheticEEGDataset('/path/to/database.lmdb')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Train
for batch_idx, (data, labels) in enumerate(dataloader):
    # data: (batch_size, n_channels, 4, 200)
    # labels: (batch_size, 1)
    ...
```

## 6. 예상 데이터 크기

| 피험자 수 | Task | 예상 샘플 수 | 예상 LMDB 크기 |
|----------|------|-------------|---------------|
| 10 | Motor Imagery | ~240 | ~100 MB |
| 50 | Motor Imagery | ~1,200 | ~500 MB |
| 100 | Motor Imagery | ~2,400 | ~1 GB |
| 100 | All tasks | ~15,000 | ~6-8 GB |

## 7. 트러블슈팅

### LMDB map_size 에러
```
Error: mdb_txn_begin: MDB_MAP_FULL: Environment mapsize limit reached
```

**해결:** `--map_size` 증가
```bash
python preproc_synthetic_to_lmdb.py \
    --map_size 100000000000  # 100 GB
```

### 메모리 부족
```
MemoryError: Unable to allocate array
```

**해결:** Task별로 나누어 처리
```bash
# Motor Imagery만 먼저
python preproc_synthetic_to_lmdb.py \
    --tasks motor_imagery \
    --output_lmdb motor_imagery.lmdb

# 그 다음 P300
python preproc_synthetic_to_lmdb.py \
    --tasks p300 \
    --output_lmdb p300.lmdb
```

### Channel 수 불일치
```
ValueError: channel count mismatch
```

**해결:** 생성 시 사용한 채널 수와 전처리 시 채널 수 확인
- 기본: 64 channels
- 필요시 코드에서 `n_channels` 수정

## 8. 참고: PhysioNet 전처리와의 차이점

| 항목 | PhysioNet (Real) | Synthetic |
|-----|------------------|-----------|
| Line noise | 60 Hz notch filter 필요 | 필요 없음 |
| Artifacts | ICA, artifact rejection | 필요 없음 |
| Bad channels | 실제 손상 채널 | 마킹만 됨 |
| High-pass filter | 0.3 Hz | 선택적 |
| Epoching | Event 기반 | Task별 다름 |

## 9. 다음 단계

1. ✅ Multi-subject 데이터 생성
2. ✅ LMDB 변환
3. ✅ LMDB 검증
4. → **모델 학습에 사용**

LMDB를 기존 모델의 데이터 경로로 지정하면 바로 사용 가능합니다!

```python
# 기존 코드에서
dataset = EEGDataset(lmdb_path='/path/to/PhysioNet.lmdb')

# Synthetic 데이터로 변경
dataset = EEGDataset(lmdb_path='/path/to/Synthetic_MI.lmdb')
```

## 10. 문의

문제가 발생하면:
1. `verify_lmdb.py`로 데이터 확인
2. 로그 파일 확인
3. 샘플 데이터 직접 로드해서 shape 확인
