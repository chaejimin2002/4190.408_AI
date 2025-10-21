# HOG 기반 얼굴 검출 (Face Detection using HOG)

HOG(Histogram of Oriented Gradients) 특징 추출 방식과 NCC(Normalized Cross Correlation)를 이용한 얼굴 검출 시스템 구현

## 프로젝트 개요

본 프로젝트는 HOG 특징을 추출하고 템플릿 매칭을 통해 이미지에서 얼굴을 검출하는 시스템입니다. NMS(Non-Maximum Suppression)를 통해 중복 검출을 제거하여 정확한 얼굴 위치를 찾아냅니다.

## 환경 설정

### 필요 라이브러리
```bash
pip install opencv-python numpy matplotlib
```

### 실행 방법
```bash
python src/HOG_ver1.py
```

## 구현 기능

### 1. 미분 필터 생성 (`get_differential_filter`)
- **입력**: 없음
- **출력**: x방향 필터, y방향 필터 (각 3x3)
- **설명**: 이미지의 x, y 방향 그래디언트를 계산하기 위한 Sobel 필터 생성

```python
filter_x = [[1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]]

filter_y = [[-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]]
```

### 2. 이미지 필터링 (`filter_image`)
- **입력**: 
  - `im`: 2D 이미지 (m × n)
  - `filter`: 2D 필터 (k × k)
- **출력**: 필터링된 이미지 (m × n)
- **설명**: 제로 패딩을 적용한 후 컨볼루션 연산 수행

### 3. 그래디언트 계산 (`get_gradient`)
- **입력**: 
  - `im_dx`: x방향 미분 이미지
  - `im_dy`: y방향 미분 이미지
- **출력**: 
  - `grad_mag`: 그래디언트 크기
  - `grad_angle`: 그래디언트 방향 (0 ~ π)
- **설명**: 
  - 크기: √(dx² + dy²)
  - 방향: arctan2(dy, dx), 음수 각도는 π를 더해 0~π 범위로 정규화

### 4. 히스토그램 구축 (`build_histogram`)
- **입력**: 
  - `grad_mag`: 그래디언트 크기
  - `grad_angle`: 그래디언트 방향
  - `cell_size`: 셀 크기
- **출력**: 방향 히스토그램 (M × N × 6)
- **설명**: 
  - 이미지를 cell_size × cell_size 크기의 셀로 분할
  - 각 셀에서 6개의 방향 빈으로 히스토그램 생성
  - 방향 빈 범위:
    - Bin 0: [0, π/12) ∪ [11π/12, π)
    - Bin 1: [π/12, π/4)
    - Bin 2: [π/4, 5π/12)
    - Bin 3: [5π/12, 7π/12)
    - Bin 4: [7π/12, 3π/4)
    - Bin 5: [3π/4, 11π/12)

### 5. 블록 디스크립터 생성 (`get_block_descriptor`)
- **입력**: 
  - `ori_histo`: 방향 히스토그램 (M × N × 6)
  - `block_size`: 블록 크기 (셀 단위)
- **출력**: 정규화된 히스토그램 ((M-block_size+1) × (N-block_size+1) × (6×block_size²))
- **설명**: 
  - block_size × block_size 셀을 하나의 블록으로 묶음
  - 각 블록의 히스토그램을 L2 정규화: v / √(‖v‖² + ε²), ε = 10⁻³

### 6. HOG 특징 추출 (`extract_hog`)
- **입력**: 
  - `im`: 2D 이미지
  - `visualize`: 시각화 여부 (기본값: False)
  - `cell_size`: 셀 크기 (기본값: 8)
  - `block_size`: 블록 크기 (기본값: 2)
- **출력**: 1D HOG 특징 벡터
- **처리 과정**:
  1. 이미지 정규화 (0~1 범위)
  2. 미분 필터 적용
  3. 그래디언트 계산
  4. 히스토그램 구축
  5. 블록 디스크립터 생성 및 평탄화

### 7. NCC (Normalized Cross Correlation)
- **입력**: 
  - `hog_target`: 대상 HOG 특징
  - `hog_template`: 템플릿 HOG 특징
- **출력**: 유사도 점수 (-1 ~ 1)
- **설명**: 평균 제거 후 정규화된 내적 계산

### 8. IoU (Intersection over Union)
- **입력**: 
  - `box1`, `box2`: 바운딩 박스 [x, y, score]
  - `box_size`: 박스 크기 [width, height]
- **출력**: IoU 값 (0 ~ 1)
- **설명**: 두 박스의 겹침 정도 계산

### 9. NMS (Non-Maximum Suppression)
- **입력**: 
  - `bounding_boxes`: 검출된 박스들 (k × 3)
  - `box_size`: 박스 크기
  - `iou_threshold`: IoU 임계값 (기본값: 0.5)
- **출력**: 필터링된 박스들
- **설명**: 
  - NCC 점수 기준으로 내림차순 정렬
  - IoU가 임계값 이상인 중복 박스 제거

### 10. 얼굴 인식 (`face_recognition`)
- **입력**: 
  - `I_target`: 대상 이미지 (M × N)
  - `I_template`: 템플릿 이미지 (m × n)
- **출력**: 검출된 얼굴 바운딩 박스 (k × 3)
- **처리 과정**:
  1. 템플릿의 HOG 특징 추출
  2. 슬라이딩 윈도우로 대상 이미지 탐색
  3. 각 위치에서 HOG 특징 추출 및 NCC 계산
  4. NCC > 0.48인 위치 선택
  5. NMS로 중복 제거

### 11. 시각화 (`visualize_face_detection`)
- **입력**: 
  - `I_target`: 대상 이미지
  - `bounding_boxes`: 검출된 박스들
  - `box_size`: 박스 크기
- **출력**: 시각화 결과 저장 (`result_face_detection.png`)
- **설명**: 검출된 얼굴 영역에 사각형과 NCC 점수 표시

## 파일 구조

```
HW2/
├── src/
│   └── HOG_ver1.py          # 메인 구현 파일
├── imgsrc/                   # 입력 이미지 폴더
│   ├── cameraman.tif        # HOG 시각화 테스트 이미지
│   ├── target.png           # 얼굴 검출 대상 이미지
│   └── template.png         # 얼굴 템플릿 이미지
├── hog.png                   # HOG 시각화 결과
├── result_face_detection.png # 얼굴 검출 결과
└── README.md                 # 본 문서
```

## 주요 알고리즘

### HOG 특징 추출 파이프라인
```
입력 이미지
    ↓
정규화 (0~1)
    ↓
미분 필터 적용 (Sobel)
    ↓
그래디언트 크기 및 방향 계산
    ↓
셀 단위 방향 히스토그램 생성 (6 bins)
    ↓
블록 단위 L2 정규화
    ↓
1D 특징 벡터
```

### 얼굴 검출 파이프라인
```
대상 이미지 + 템플릿 이미지
    ↓
템플릿 HOG 특징 추출
    ↓
슬라이딩 윈도우 (모든 위치)
    ↓
각 위치의 HOG 특징 추출
    ↓
NCC로 유사도 계산
    ↓
임계값(0.48) 이상인 후보 선택
    ↓
NMS로 중복 제거 (IoU threshold: 0.5)
    ↓
최종 검출 결과
```

## 파라미터 설정

- **Cell Size**: 8×8 픽셀
- **Block Size**: 2×2 셀
- **방향 빈 개수**: 6
- **NCC 임계값**: 0.48
- **IoU 임계값**: 0.5
- **L2 정규화 ε**: 10⁻³

## 결과

- `hog.png`: HOG 특징의 시각화 (방향 벡터 표시)
- `result_face_detection.png`: 검출된 얼굴 영역과 신뢰도 점수

## 구현 특징

1. **완전한 Numpy 기반 구현**: HOG 특징 추출이 순수 Numpy로 구현되어 알고리즘 이해에 용이
2. **슬라이딩 윈도우 방식**: 모든 가능한 위치를 탐색하여 얼굴 검출
3. **효율적인 중복 제거**: NMS를 통해 중복 검출 최소화
4. **시각화 지원**: HOG 특징 및 검출 결과를 직관적으로 확인 가능

## 참고사항

- 본 구현은 교육 목적으로 작성되어 최적화보다는 가독성과 이해도에 중점을 둠
- 실제 응용을 위해서는 스케일 변화, 회전 등의 추가적인 고려사항이 필요
- 더 빠른 처리를 위해서는 이미지 피라미드, 적분 이미지 등의 기법 활용 가능

