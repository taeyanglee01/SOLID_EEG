# Multi-Subject Task-Specific EEG Data Generation

이 스크립트는 여러 피험자(subject)에 대한 task-specific synthetic EEG 데이터를 생성합니다.

## 주요 특징

- **Multiple Subjects**: 원하는 수만큼의 피험자 데이터를 생성
- **Subject-specific variations**: 각 피험자마다 진폭 및 주파수 변동을 적용하여 개인차를 시뮬레이션
- **Organized directory structure**: 각 피험자별로 별도의 서브 디렉토리 생성 (`sub-001`, `sub-002`, ...)
- **All task types**: Motor Imagery, Seizure, P300 Oddball, Emotion 등 모든 태스크 지원

## 디렉토리 구조

```
output_dir/
├── sub-001/
│   ├── motor_imagery_left_hand_raw.fif
│   ├── motor_imagery_left_hand_groundtruth_raw.fif
│   ├── motor_imagery_left_hand_events.txt
│   ├── motor_imagery_right_hand_raw.fif
│   ├── ...
│   ├── seizure_absence_raw.fif
│   ├── ...
│   ├── p300_oddball_raw.fif
│   ├── ...
│   └── emotion_positive_raw.fif
├── sub-002/
│   └── ...
├── sub-003/
│   └── ...
└── ...
```

## 사용법

### 기본 사용 (모든 태스크, 10명의 피험자)

```bash
python generate_task_specific_eeg_multisubject.py
```

### 피험자 수 지정

```bash
python generate_task_specific_eeg_multisubject.py --n_subjects 50
```

### 출력 디렉토리 지정

```bash
python generate_task_specific_eeg_multisubject.py \
    --n_subjects 30 \
    --output_dir /pscratch/sd/t/tylee/SOLID_EEG_RESULT/synthetic_eeg/multisubject_data
```

### 특정 태스크만 생성

```bash
# Motor Imagery만 생성
python generate_task_specific_eeg_multisubject.py \
    --n_subjects 20 \
    --tasks motor

# P300과 Emotion만 생성
python generate_task_specific_eeg_multisubject.py \
    --n_subjects 15 \
    --tasks p300,emotion
```

### 데이터 길이 조정

```bash
python generate_task_specific_eeg_multisubject.py \
    --n_subjects 10 \
    --motor_duration 120 \
    --seizure_duration 90 \
    --p300_duration 600 \
    --emotion_duration 180
```

### 채널 수 및 샘플링 주파수 조정

```bash
python generate_task_specific_eeg_multisubject.py \
    --n_subjects 10 \
    --n_channels 32 \
    --sfreq 500
```

## 명령줄 인자 (Arguments)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--n_subjects` | 10 | 생성할 피험자 수 |
| `--output_dir` | `/pscratch/sd/t/tylee/SOLID_EEG_RESULT/synthetic_eeg/multisubject_task_data` | 출력 디렉토리 경로 |
| `--n_channels` | 64 | EEG 채널 수 |
| `--sfreq` | 250.0 | 샘플링 주파수 (Hz) |
| `--tasks` | all | 생성할 태스크: `all`, `motor`, `seizure`, `p300`, `emotion` 또는 콤마로 구분된 리스트 |
| `--motor_duration` | 60 | Motor imagery 태스크 길이 (초) |
| `--seizure_duration` | 60 | Seizure 데이터 길이 (초) |
| `--p300_duration` | 300 | P300 oddball 태스크 길이 (초) |
| `--emotion_duration` | 120 | Emotion 태스크 길이 (초) |

## 생성되는 태스크

### 1. Motor Imagery (4개 조건)
- `left_hand`: 왼손 운동 상상
- `right_hand`: 오른손 운동 상상
- `both_hands`: 양손 운동 상상
- `feet`: 발 운동 상상

### 2. Seizure (3개 타입)
- `absence`: Generalized 3Hz spike-wave discharge
- `focal_frontal`: Frontal focal seizure with propagation
- `focal_temporal`: Temporal focal seizure with propagation

### 3. P300 Oddball (1개)
- Event-related potential with target/non-target stimuli

### 4. Emotion (3개 조건)
- `positive`: Positive emotional state (left frontal activation)
- `negative`: Negative emotional state (right frontal activation)
- `neutral`: Neutral emotional state

## 피험자별 변동 (Subject Variations)

각 피험자마다 다음과 같은 개인차가 적용됩니다:

1. **Amplitude scaling**: 0.8 ~ 1.2 (±20%)
2. **Frequency shift**: 0.95 ~ 1.05 (±5%)

이러한 변동은 실제 EEG 데이터의 개인차를 시뮬레이션합니다.

## 출력 파일

각 태스크마다 다음 파일들이 생성됩니다:

1. **`*_raw.fif`**: Bad channels가 포함된 데이터 (interpolation 테스트용)
2. **`*_groundtruth_raw.fif`**: Bad channels가 없는 ground truth 데이터
3. **`*_events.txt`**: Event markers (해당되는 경우)

## 예시: 대규모 데이터 생성

```bash
# 100명의 피험자, 모든 태스크, 긴 duration
python generate_task_specific_eeg_multisubject.py \
    --n_subjects 100 \
    --output_dir /pscratch/sd/t/tylee/SOLID_EEG_RESULT/synthetic_eeg/large_cohort \
    --motor_duration 120 \
    --seizure_duration 120 \
    --p300_duration 600 \
    --emotion_duration 240
```

## SLURM 스크립트 예시

서버에서 대량의 데이터를 생성할 때 사용할 수 있는 SLURM 스크립트:

```bash
#!/bin/bash
#SBATCH --job-name=generate_multisubject_eeg
#SBATCH --output=generate_multisubject_eeg_%j.out
#SBATCH --error=generate_multisubject_eeg_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

# Activate conda environment
source activate your_eeg_env

# Run the script
python generate_task_specific_eeg_multisubject.py \
    --n_subjects 100 \
    --output_dir /pscratch/sd/t/tylee/SOLID_EEG_RESULT/synthetic_eeg/multisubject_data \
    --motor_duration 120 \
    --seizure_duration 120 \
    --p300_duration 600 \
    --emotion_duration 240

echo "Data generation complete!"
```

## 의존성 (Dependencies)

- numpy
- mne
- scipy
- pathlib
- argparse

원본 `generate_task_specific_eeg.py` 파일과 `generate_realistic_eeg.py` 파일이 같은 디렉토리에 있어야 합니다.

## 주의사항

1. 대량의 데이터를 생성할 경우 충분한 디스크 공간이 필요합니다.
   - 피험자당 약 100-500MB (태스크 수와 duration에 따라 다름)

2. 생성 시간은 피험자 수, 태스크 수, duration에 비례합니다.
   - 예상 시간: 피험자당 약 1-3분

3. 재현성을 위해 각 피험자의 variation은 고정된 시드를 사용하지만,
   실제 데이터 생성은 매번 다른 랜덤 시드를 사용합니다.

## 기존 코드와의 차이점

- **기존 (`generate_task_specific_eeg.py`)**: 단일 피험자, 모든 파일이 하나의 디렉토리에 저장
- **새 버전 (`generate_task_specific_eeg_multisubject.py`)**:
  - 다중 피험자 지원
  - 피험자별 서브 디렉토리 생성
  - 피험자별 변동 적용
  - 명령줄 인자로 유연한 설정 가능
  - 태스크 선택 기능

## 문제 해결

### ImportError 발생 시

```bash
# 현재 디렉토리를 PYTHONPATH에 추가
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python generate_task_specific_eeg_multisubject.py
```

### 메모리 부족 시

피험자를 여러 배치로 나누어 생성하거나, duration을 줄이거나, 한 번에 생성하는 태스크 수를 줄이세요.

```bash
# 10명씩 나누어 생성
for i in {0..9}; do
    start=$((i * 10 + 1))
    python generate_task_specific_eeg_multisubject.py \
        --n_subjects 10 \
        --output_dir /path/to/output/batch_${i}
done
```
