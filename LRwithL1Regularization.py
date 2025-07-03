import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

class L1LogisticRegression:
    """L1 정규화 로지스틱 회귀분석 클래스 (단계별 디버깅 버전)"""
    
    def __init__(self, target_col='암종', random_state=42):
        self.target_col = target_col
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = None
        self.results = {}
        
        # 단계별 데이터 저장
        self.raw_data = None
        self.clean_data = None
        self.X_processed = None
        self.y_encoded = None
        self.best_c = None
        self.encoder = None  # LabelEncoderWithNA 저장
        
        print("🎯 L1 로지스틱 회귀분석 객체 생성됨")
        print(f"   타겟 변수: {self.target_col}")
        print(f"   랜덤 시드: {self.random_state}")
        print("   💡 LabelEncoderWithNA + fillna(0) 처리된 데이터를 위한 특별 버전")
    
    def step1_check_data(self, df, encoder=None):
        """
        1단계: 데이터 상태를 확인합니다.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            LabelEncoderWithNA로 인코딩된 데이터프레임
        encoder : LabelEncoderWithNA
            원본 값으로 변환하기 위한 인코더 (선택사항)
        """
        print("\n🔍 1단계: 데이터 상태 확인")
        print("=" * 50)
        
        self.raw_data = df.copy()
        self.encoder = encoder
        
        # 기본 정보
        print(f"📊 데이터프레임 크기: {df.shape}")
        print(f"📊 타겟 변수: {self.target_col}")
        
        # 타겟 변수 분포 확인
        if self.target_col in df.columns:
            target_values = df[self.target_col].dropna()
            target_counts = target_values.value_counts().sort_index()
            
            print(f"\n🎯 {self.target_col} 분포:")
            print("-" * 40)
            
            # 인코더가 있으면 원본 값으로 변환
            if encoder and hasattr(encoder, 'get_mapping'):
                target_mapping = encoder.get_mapping(self.target_col)
                inverse_mapping = {v: k for k, v in target_mapping.items()}
                
                for encoded_val, count in target_counts.items():
                    percentage = (count / len(target_values)) * 100
                    original_val = inverse_mapping.get(encoded_val, f"Unknown({encoded_val})")
                    if original_val == 'NA':
                        original_val = '결측값'
                    print(f"  {encoded_val} → {original_val}: {count}개 ({percentage:.1f}%)")
            else:
                for encoded_val, count in target_counts.items():
                    percentage = (count / len(target_values)) * 100
                    print(f"  {encoded_val}: {count}개 ({percentage:.1f}%)")
            
            # 클래스 불균형 체크
            min_class_ratio = target_counts.min() / target_counts.max()
            if min_class_ratio < 0.1:
                print(f"⚠️  클래스 불균형 주의! 최소/최대 비율: {min_class_ratio:.3f}")
            else:
                print(f"✅ 클래스 균형 양호. 최소/최대 비율: {min_class_ratio:.3f}")
        else:
            print(f"❌ 타겟 변수 '{self.target_col}'이 데이터에 없습니다!")
            return None
        
        # 결측값 확인
        print(f"\n🔍 결측값 현황:")
        print("-" * 30)
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]
        
        if len(missing_cols) > 0:
            print("결측값이 있는 컬럼:")
            for col, count in missing_cols.head(10).items():  # 상위 10개만 표시
                percentage = (count / len(df)) * 100
                print(f"  {col}: {count}개 ({percentage:.1f}%)")
            
            if len(missing_cols) > 10:
                print(f"  ... 및 {len(missing_cols)-10}개 컬럼 더")
            
            # 결측값 제거 후 크기
            df_clean = df.dropna()
            removed_rows = len(df) - len(df_clean)
            print(f"\n📉 결측값 제거 후: {df_clean.shape} (제거된 행: {removed_rows})")
            
            if removed_rows / len(df) > 0.5:
                print("⚠️  데이터의 50% 이상이 결측값으로 제거됩니다!")
        else:
            print("  ✅ 결측값 없음")
            df_clean = df.copy()
        
        # 데이터 타입과 고유값 확인 (LabelEncoderWithNA 결과 고려)
        print(f"\n📋 데이터 타입 및 인코딩 현황:")
        print("-" * 50)
        
        # 인코더 정보 확인
        if encoder and hasattr(encoder, 'get_mapping'):
            encoded_columns = list(encoder.get_mapping().keys())
            print(f"📊 인코딩된 컬럼: {len(encoded_columns)}개")
            if len(encoded_columns) > 0:
                print(f"   예시: {encoded_columns[:5]}")
        
        # 혼재된 타입 문제 체크 (fillna(0) 후 발생)
        mixed_type_columns = []
        numeric_columns = []
        
        for col in df.columns[:20]:  # 상위 20개 컬럼만 체크
            unique_vals = df[col].dropna().unique()
            
            # 타입 혼재 체크
            has_numeric = any(isinstance(x, (int, float)) and not isinstance(x, bool) for x in unique_vals)
            has_string = any(isinstance(x, str) for x in unique_vals)
            
            if has_numeric and has_string:
                mixed_type_columns.append(col)
                print(f"⚠️  {col}: 타입 혼재 - {list(unique_vals[:5])} ...")
            elif has_numeric:
                numeric_columns.append(col)
                unique_count = len(unique_vals)
                
                # 인코딩된 값과 원본 값 표시
                if encoder and hasattr(encoder, 'get_mapping') and col in encoder.get_mapping():
                    mapping = encoder.get_mapping(col)
                    inverse_mapping = {v: k for k, v in mapping.items()}
                    
                    if unique_count <= 10:
                        original_vals = []
                        for val in sorted(unique_vals[:5]):
                            orig_val = inverse_mapping.get(val, f"Unknown({val})")
                            if orig_val == 'NA':
                                orig_val = '결측값'
                            original_vals.append(f"{val}→{orig_val}")
                        print(f"  {col}: {unique_count}개 고유값 → {original_vals}")
                    else:
                        print(f"  {col}: {unique_count}개 고유값 (인코딩됨)")
                else:
                    if unique_count <= 10:
                        print(f"  {col}: {unique_count}개 → {list(unique_vals[:5])}")
                    else:
                        print(f"  {col}: {unique_count}개 고유값")
        
        print(f"\n📊 데이터 타입 요약:")
        print(f"  수치형 컬럼: {len(numeric_columns)}개")
        print(f"  타입 혼재: {len(mixed_type_columns)}개")
        
        if mixed_type_columns:
            print(f"\n⚠️  타입 혼재 컬럼 발견: {mixed_type_columns[:5]}")
            print("   fillna(0) 처리로 인한 것으로 추정됩니다.")
            print("   전처리에서 타입 통일이 필요합니다.")
        
        self.clean_data = df_clean
        print(f"\n✅ 1단계 완료. 정제된 데이터: {df_clean.shape}")
        return df_clean
    
    def step2_prepare_data(self, feature_cols=None, numeric_cols=None):
        """
        2단계: 데이터 전처리를 수행합니다.
        LabelEncoderWithNA + fillna(0) 처리된 데이터를 고려하여 
        수치형 변수와 카테고리형 변수를 구분 처리합니다.
        
        Parameters:
        -----------
        feature_cols : list
            사용할 feature 컬럼 리스트
        numeric_cols : list
            수치형 변수로 처리할 컬럼 리스트 (CBC, 나이, 키, 몸무게 등)
        """
        print("\n🔄 2단계: 데이터 전처리 (수치형/카테고리형 구분)")
        print("=" * 50)
        
        if self.clean_data is None:
            print("❌ step1_check_data()를 먼저 실행하세요!")
            return None, None
        
        df = self.clean_data.copy()
        
        # 기본 수치형 변수 정의 (CBC + 인구통계학적 수치 변수)
        default_numeric_cols = [
            # CBC 데이터
            'WBC', 'RBC', 'Hb', 'Hct', 'MCV', 'MCH', 'MCHC', 'RDW', 
            'PLT', 'PCT', 'MPV', 'PDW', 'ESR', 'Myelocyte', 'Metamyelocyte',
            'Band neutrophil', 'Segmented neutrophil', 'Lymphocyte', 'Monocyte',
            'Eosinophil', 'Basophil', 'Blast', 'Promonocyte', 'Promyelocyte',
            'Immature cell', 'Atypical lymphocyte', 'Normoblast', 'ANC',
            # 인구통계학적 수치 변수  
            'AGE', 'Height', 'Weight', 'BMI'
        ]
        
        if numeric_cols is None:
            numeric_cols = default_numeric_cols
        
        # feature 컬럼 설정
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != self.target_col]
            print(f"📊 모든 컬럼을 feature로 사용: {len(feature_cols)}개")
        else:
            # 존재하는 컬럼만 필터링
            available_cols = [col for col in feature_cols if col in df.columns]
            missing_cols = [col for col in feature_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️  없는 컬럼 제외: {missing_cols[:5]}...")
            
            feature_cols = available_cols
            print(f"📊 사용할 feature: {len(feature_cols)}개")
        
        # 수치형과 카테고리형 분리
        available_numeric_cols = [col for col in numeric_cols if col in feature_cols]
        categorical_feature_cols = [col for col in feature_cols if col not in numeric_cols]
        
        print(f"📊 변수 타입 분류:")
        print(f"  수치형 변수: {len(available_numeric_cols)}개")
        print(f"  카테고리형 변수: {len(categorical_feature_cols)}개")
        
        # 수치형 변수 처리
        print(f"\n🔢 수치형 변수 처리 중...")
        numeric_data = df[available_numeric_cols].copy()
        
        for col in available_numeric_cols:
            # 문자열이 섞여있으면 숫자로 변환
            if col in df.columns:
                original_type = numeric_data[col].dtype
                # 강제로 숫자 변환
                numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
                # NaN을 0으로 채움
                numeric_data[col] = numeric_data[col].fillna(0)
                
                if original_type == 'object':
                    print(f"  {col}: {original_type} → 수치형 변환")
        
        print(f"  수치형 데이터 형태: {numeric_data.shape}")
        
        # 카테고리형 변수 처리 (타입 혼재 해결)
        print(f"\n🏷️  카테고리형 변수 처리 중...")
        categorical_data = df[categorical_feature_cols].copy()
        
        for col in categorical_feature_cols:
            if col in df.columns:
                # 타입 혼재 해결: 모든 값을 문자열로 통일
                unique_vals = categorical_data[col].dropna().unique()
                has_numeric = any(isinstance(x, (int, float)) and not isinstance(x, bool) for x in unique_vals)
                has_string = any(isinstance(x, str) for x in unique_vals)
                
                if has_numeric and has_string:
                    # 숫자 0을 문자열 '0'으로 통일
                    categorical_data[col] = categorical_data[col].astype(str)
                    categorical_data[col] = categorical_data[col].replace(['0.0', '0'], '0')
                    print(f"  {col}: 문자열로 통일 변환 (0 → '0')")
                elif has_numeric:
                    # 모두 숫자인 경우 문자열로 변환
                    categorical_data[col] = categorical_data[col].astype(str)
                    categorical_data[col] = categorical_data[col].replace(['0.0'], '0')
                    print(f"  {col}: 문자열로 변환")
        
        # NaN을 '0'으로 통일
        if categorical_data.isnull().any().any():
            categorical_data = categorical_data.fillna('0')
        
        print(f"  카테고리형 데이터 형태: {categorical_data.shape}")
        
        # 카테고리형 변수 원-핫 인코딩
        if len(categorical_feature_cols) > 0:
            print(f"\n🔄 카테고리형 변수 원-핫 인코딩 중...")
            categorical_encoded = pd.get_dummies(categorical_data, drop_first=True)
            print(f"  원-핫 인코딩 후: {categorical_encoded.shape}")
        else:
            categorical_encoded = pd.DataFrame(index=df.index)
        
        # 수치형과 카테고리형 결합
        print(f"\n🔗 수치형과 카테고리형 데이터 결합 중...")
        X_encoded = pd.concat([numeric_data, categorical_encoded], axis=1)
        
        self.feature_names = X_encoded.columns.tolist()
        print(f"📊 최종 feature 수: {len(self.feature_names)}개")
        print(f"  - 수치형: {len(available_numeric_cols)}개")
        print(f"  - 카테고리형 (원-핫): {len(categorical_encoded.columns)}개")
        
        # 타겟 변수 처리
        y = df[self.target_col].copy()
        
        # 타겟 변수 처리 (LabelEncoderWithNA 고려)
        print(f"\n🎯 타겟 변수 전처리...")
        
        # 타겟 변수도 문자열로 통일 (만약 혼재되어 있다면)
        target_unique = y.dropna().unique()
        has_numeric = any(isinstance(x, (int, float)) and not isinstance(x, bool) for x in target_unique)
        has_string = any(isinstance(x, str) for x in target_unique)
        
        if has_numeric and has_string:
            y = y.astype(str)
            y = y.replace(['0.0', '0'], '0')
            print("  타겟 변수: 문자열로 통일 변환")
        elif has_numeric:
            y = y.astype(str)
            y = y.replace(['0.0'], '0')
            print("  타겟 변수: 문자열로 변환")
        
        unique_targets = sorted(y.unique())
        print(f"   타겟 고유값: {unique_targets}")
        
        # LabelEncoderWithNA의 매핑을 활용하여 sklearn 호환 처리
        if self.encoder and hasattr(self.encoder, 'get_mapping'):
            target_mapping = self.encoder.get_mapping(self.target_col)
            if target_mapping:
                print(f"   원본 매핑: {target_mapping}")
                
                # 문자열로 변환된 매핑 생성
                str_to_original = {}
                for original, encoded in target_mapping.items():
                    str_encoded = str(encoded)
                    str_to_original[str_encoded] = original
                
                # 유효한 클래스만 선택 ('0'은 결측값이므로 제외)
                valid_indices = y[y != '0'].index
                y_valid = y.loc[valid_indices]
                X_encoded = X_encoded.loc[valid_indices]
                
                # 원본 값으로 변환 후 sklearn 인코딩
                y_original = [str_to_original.get(str_val, 'Unknown') for str_val in y_valid]
                original_classes = sorted(list(set(y_original)))
                original_classes = [cls for cls in original_classes if cls != 'NA' and cls != 'Unknown']
                
                self.label_encoder.classes_ = np.array(original_classes)
                y_encoded = self.label_encoder.transform(y_original)
                
                print(f"   유효한 클래스: {original_classes}")
                print(f"   결측값 제거 후 데이터 크기: X={X_encoded.shape}, y={len(y_encoded)}")
            else:
                print("⚠️  타겟 변수 매핑 정보가 없습니다.")
                y_valid = y[y != '0']
                X_encoded = X_encoded.loc[y_valid.index]
                y_encoded = self.label_encoder.fit_transform(y_valid)
        else:
            print("⚠️  LabelEncoderWithNA 정보가 없어 기본 처리합니다.")
            y_valid = y[y != '0']
            X_encoded = X_encoded.loc[y_valid.index]
            y_encoded = self.label_encoder.fit_transform(y_valid)
        
        print(f"\n📊 최종 타겟 클래스: {len(self.label_encoder.classes_)}개")
        print(f"   클래스명: {list(self.label_encoder.classes_)}")
        
        # 최종 확인
        print(f"\n✅ 2단계 완료:")
        print(f"   X shape: {X_encoded.shape}")
        print(f"   y shape: {y_encoded.shape}")
        print(f"   X 데이터 타입: {X_encoded.dtypes.value_counts().to_dict()}")
        
        self.X_processed = X_encoded
        self.y_encoded = y_encoded
        
        return X_encoded, y_encoded
    
    def step3_explain_c_parameter(self):
        """
        3단계: C 파라미터의 의미를 설명합니다.
        """
        print("\n📚 3단계: C 파라미터 이해하기")
        print("=" * 50)
        print("L1 로지스틱 회귀에서 C는 '정규화 강도의 역수'입니다.")
        print()
        print("🔹 C가 클수록 (예: C=100)")
        print("   → 정규화 약함 → 더 많은 특성 사용 → 복잡한 모델")
        print("   → 과적합 위험 증가, 하지만 훈련 데이터에 잘 맞음")
        print()
        print("🔹 C가 작을수록 (예: C=0.01)")
        print("   → 정규화 강함 → 적은 특성 사용 → 간단한 모델") 
        print("   → 언더피팅 위험 증가, 하지만 일반화 성능 좋음")
        print()
        print("🎯 목표: 적절한 C 값으로 편향-분산 트레이드오프 최적화!")
        print()
        print("📐 수식:")
        print("   Cost = 로지스틱 손실 + (1/C) × Σ|계수|")
        print("   여기서 Σ|계수|가 L1 정규화 항 (Lasso)")
        print()
        print("💡 L1 정규화의 특징:")
        print("   - 일부 계수를 정확히 0으로 만듦 → 자동 특성 선택")
        print("   - 중요하지 않은 변수들이 모델에서 제거됨")
        print("   - 해석하기 쉬운 sparse한 모델 생성")
    
    def step4_find_optimal_c(self, c_range=None, cv_folds=5):
        """
        4단계: 교차검증으로 최적 C 값을 찾습니다.
        """
        print("\n🎯 4단계: 최적 C 값 탐색")
        print("=" * 40)
        
        if self.X_processed is None or self.y_encoded is None:
            print("❌ step2_prepare_data()를 먼저 실행하세요!")
            return None
        
        X, y = self.X_processed, self.y_encoded
        
        if c_range is None:
            c_range = [0.001, 0.01, 0.1, 1, 10]
        
        print(f"테스트할 C 값들: {c_range}")
        print(f"교차검증 fold: {cv_folds}")
        print(f"데이터 크기: {X.shape}")
        print()
        
        # 데이터 스케일링 (매우 중요!)
        X_scaled = self.scaler.fit_transform(X)
        print("✅ 데이터 스케일링 완료")
        print("   (로지스틱 회귀는 특성 스케일에 민감하므로 반드시 필요)")
        
        # 각 C 값에 대해 교차검증 수행
        results = []
        
        print("\n📊 C 값별 교차검증 결과:")
        print("-" * 55)
        print("    C 값    |  평균 정확도  |  표준편차  |  특성 수")
        print("-" * 55)
        
        for C in c_range:
            # L1 정규화 로지스틱 회귀 모델
            model = LogisticRegression(
                penalty='l1',           # L1 정규화 (Lasso)
                C=C,                   # 정규화 강도
                solver='liblinear',    # L1에 적합한 solver
                random_state=self.random_state,
                max_iter=1000
            )
            
            # 교차검증 수행
            cv_scores = cross_val_score(
                model, X_scaled, y,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                 random_state=self.random_state),
                scoring='accuracy'
            )
            
            # 특성 선택 개수 확인 (전체 데이터로 학습)
            model.fit(X_scaled, y)
            selected_features = np.sum(np.abs(model.coef_) > 1e-6, axis=1).sum()
            
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            results.append({
                'C': C,
                'mean_accuracy': mean_score,
                'std_accuracy': std_score,
                'cv_scores': cv_scores,
                'selected_features': selected_features
            })
            
            print(f"  {C:8.3f}  |   {mean_score:.4f}   |  {std_score:.4f}  |   {selected_features:3d}")
        
        print("-" * 55)
        
        # 최적 C 값 선택
        best_result = max(results, key=lambda x: x['mean_accuracy'])
        self.best_c = best_result['C']
        best_score = best_result['mean_accuracy']
        
        print(f"\n🏆 최적 결과:")
        print(f"   최적 C: {self.best_c}")
        print(f"   최고 정확도: {best_score:.4f} (±{best_result['std_accuracy']:.4f})")
        print(f"   선택된 특성 수: {best_result['selected_features']}개 / {len(self.feature_names)}개")
        
        # 시각화
        self._plot_c_validation(results)
        
        return self.best_c, results
    
    def _plot_c_validation(self, results):
        """C 값 검증 결과를 시각화합니다."""
        
        c_values = [r['C'] for r in results]
        mean_scores = [r['mean_accuracy'] for r in results]
        std_scores = [r['std_accuracy'] for r in results]
        feature_counts = [r['selected_features'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 상단: 정확도 그래프
        ax1.errorbar(range(len(c_values)), mean_scores, yerr=std_scores,
                    marker='o', markersize=8, capsize=5, capthick=2, linewidth=2, color='blue')
        
        # 최적값 표시
        best_idx = c_values.index(self.best_c)
        ax1.scatter(best_idx, mean_scores[best_idx], color='red', s=150, zorder=5, 
                   label=f'최적 C = {self.best_c}')
        
        ax1.set_xlabel('C 값')
        ax1.set_ylabel('교차검증 정확도')
        ax1.set_title('C 값에 따른 모델 성능')
        ax1.set_xticks(range(len(c_values)))
        ax1.set_xticklabels([str(c) for c in c_values])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 하단: 선택된 특성 수
        ax2.bar(range(len(c_values)), feature_counts, alpha=0.7, color='green')
        ax2.set_xlabel('C 값')
        ax2.set_ylabel('선택된 특성 수')
        ax2.set_title('C 값에 따른 특성 선택 (L1 정규화 효과)')
        ax2.set_xticks(range(len(c_values)))
        ax2.set_xticklabels([str(c) for c in c_values])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 최적값 표시
        ax2.bar(best_idx, feature_counts[best_idx], color='red', alpha=0.8,
               label=f'최적 C = {self.best_c}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('c_validation_results.png', dpi=300)
        plt.show()
        
    
    def get_summary(self):
        """현재까지의 진행 상황을 요약합니다."""
        print("\n📋 진행 상황 요약")
        print("=" * 40)
        
        if self.raw_data is not None:
            print(f"✅ 1단계: 원본 데이터 {self.raw_data.shape}")
        
        if self.clean_data is not None:
            print(f"✅ 1단계: 정제 데이터 {self.clean_data.shape}")
        
        if self.X_processed is not None:
            print(f"✅ 2단계: 전처리 완료 {self.X_processed.shape}")
            print(f"         특성 개수: {len(self.feature_names)}")
            
        if self.best_c is not None:
            print(f"✅ 4단계: 최적 C = {self.best_c}")
        
        print(f"\n다음 단계:")
        if self.raw_data is None:
            print("  → step1_check_data() 실행")
        elif self.X_processed is None:
            print("  → step2_prepare_data() 실행")
        elif self.best_c is None:
            print("  → step3_explain_c_parameter() 확인")
            print("  → step4_find_optimal_c() 실행")
        else:
            print("  → 모델 학습 및 평가 준비 완료!")

# 사용 예시
if __name__ == "__main__":
    print("💡 단계별 사용법 (LabelEncoderWithNA + fillna(0) 데이터용):")
    print("=" * 60)
    print("# 1. 객체 생성")
    print("model = L1LogisticRegression(target_col='암종')")
    print()
    print("# 2. 단계별 실행")
    print("model.step1_check_data(processed_encoded, encoder=encoder)")
    print("model.step2_prepare_data(feature_cols=feature_vars)  # 또는 None")
    print("model.step3_explain_c_parameter()")
    print("model.step4_find_optimal_c()")
    print()
    print("# 3. 진행 상황 확인")
    print("model.get_summary()")
    print()
    print("🎯 주요 개선사항:")
    print("  - LabelEncoderWithNA의 inverse_transform 활용")
    print("  - fillna(0)로 인한 타입 혼재 문제 해결")
    print("  - 인코딩된 값을 원본 값으로 표시")
    print("  - 타입 통일 및 안정적인 전처리")