# 🚗 사고 차량 견적 산출 시스템 (Vehicle Repair Cost Prediction)
### YOLOv8 객체 탐지, ResNet 손상 분류 및 XGBoost 회귀 분석을 활용한 자동화 파이프라인


## 프로젝트 개요 (Overview)
본 프로젝트는 사고 차량의 이미지를 입력받아 **파손 부위 탐지**, **손상 유형 분류 및 수리 방법 예측**, 그리고 **최종 수리비 예측**까지 수행하는 End-to-End 딥러닝 파이프라인을 구축하는 것을 목표로 합니다.

### 핵심 목표
1.  차량의 주요 부품(범퍼, 도어, 램프 등)을 정확히 탐지.
2.  손상된 영역을 픽셀 단위로 분할하고, 필요한 수리 방법(교환, 판금 등)을 예측.
3.  분석된 정보를 종합하여 실제 견적서와 유사한 수준의 수리비를 산출.

---

## 방법론 (Methodology)

전체 시스템은 크게 3단계의 파이프라인으로 구성됩니다.

### 1. 부위 검출 (Part Detection) - `YOLOv8`
* **모델:** YOLOv8
* **기능:** 차량 이미지에서 34개의 차체 부위(Front bumper, Fender, Door, Lamp 등)를 탐지합니다.
* **최적화:** 미세한 파손 부위의 재현율(Recall)을 극대화하기 위해 Confidence Threshold를 **0.15**로 조정하여 견적 누락을 방지했습니다.

### 2. 손상 분석 (Damage Analysis) - `U-Net` & `ResNet34`
* **손상 영역 분할 (U-Net):** 검출된 부위 내에서 손상 영역을 픽셀 단위로 분할(Segmentation)하며, 4가지 유형(*Scratched, Crushed, Breakage, Separated*)으로 분류합니다.
* **수리 방법 예측 (ResNet34):** Multi-label 분류 모델을 사용하여 시각적 특징을 기반으로 4가지 수리 방법(*Coating, Exchange, Sheet metal, Repair*)을 복수로 예측합니다.

### 3. 수리비 예측 (Cost Prediction) - `XGBoost`
* **모델:** XGBoost Regressor
* **입력:** 탐지된 부위 정보, 예측된 수리 방법, 차종 정보 등을 조합하여 입력으로 사용합니다.
* **전처리:** `MultiLabelBinarizer`를 활용해 범주형 데이터를 인코딩하고, 비용 데이터에 로그 변환(Log Transformation)을 적용하여 예측 오차를 줄였습니다.

---

## 성능 (Performance)

### 부위 검출 (YOLO)
* **전체 mAP50:** 0.687
* **주요 부품 성능:**
    * Front Door (R): 0.995 AP
    * Front Bumper: 0.961 AP

### 수리 방법 분류 (ResNet)
* **Coating (도장):** F1-score **0.98**
* **Exchange (교환):** F1-score **0.93**

### 수리비 예측 (XGBoost)
Test 데이터셋 (7,500개) 기준:
* **RMSE:** 357,780 원
* **MAE:** 179,721 원
* **결정 계수 (R² Score):** 0.7501
* **정확도 (오차 범위 ±30%):** 68.36%

---

## 데이터셋 (Dataset)
* **출처:** AI Hub "차량 파손 이미지 데이터"
* **구성:** 사고 차량 이미지 및 라벨링 JSON (폴리곤, 손상 종류, 수리 방법, 수리비 견적 정보 포함)
* **규모:** Training (35,000+), Validation (7,500), Test (7,500)

---

## 👥 팀원 (Team Members)
| 이름 | 학번 | 역할 |
|:---:|:---:|:---|
| **전희원** | 2276274 | 수리비 예측 (Cost Prediction) / 모델 통합 |
| **공세영** | 2371006 | 손상 분류 (Damage segmenation) / 수리 방식 예측 (Repair Prediction) |
| **김문경(팀장)** | 2376029 | 부위 검출 (Part Detection) / 데이터 전처리 |

---
