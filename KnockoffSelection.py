import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

class KnockoffSelection:
    """
    Knockoff Selection for False Discovery Rate (FDR) Control
    
    이 클래스는 Model-X Knockoffs 방법을 사용하여 특성 선택에서 
    False Discovery Rate를 제어합니다.
    """
    
    def __init__(self, target_col='암종', fdr_level=0.1, random_state=42):
        self.target_col = target_col
        self.fdr_level = fdr_level  # 원하는 FDR 수준 (예: 0.1 = 10%)
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # 결과 저장
        self.X_original = None
        self.X_knockoffs = None
        self.y = None
        self.feature_names = None
        self.knockoff_stats = None
        self.selected_features = None
        self.threshold = None
        
        print("🎯 Knockoff Selection 객체 생성됨")
        print(f"   타겟 변수: {self.target_col}")
        print(f"   FDR 수준: {self.fdr_level}")
        print(f"   랜덤 시드: {self.random_state}")
    
    def step1_explain_knockoffs(self):
        """
        1단계: Knockoff Selection의 개념을 설명합니다.
        """
        print("\n📚 1단계: Knockoff Selection 이해하기")
        print("=" * 60)
        print("🔍 Knockoff Selection이란?")
        print("   변수 선택에서 False Discovery Rate(FDR)를 통계적으로 제어하는 방법")
        print()
        print("🎯 핵심 아이디어:")
        print("   1. 원본 특성 X에 대해 '가짜' 특성 X̃ (knockoff)를 생성")
        print("   2. X̃는 X와 같은 분포를 가지지만 Y와는 독립적")
        print("   3. 원본과 knockoff를 함께 모델에 넣어 중요도 비교")
        print("   4. 원본이 knockoff보다 훨씬 중요한 특성만 선택")
        print()
        print("📊 False Discovery Rate (FDR):")
        print("   FDR = E[선택된 특성 중 거짓 양성의 비율]")
        print(f"   목표: FDR ≤ {self.fdr_level} 보장")
        print()
        print("✅ 장점:")
        print("   - 통계적으로 엄격한 특성 선택")
        print("   - FDR 제어로 신뢰성 있는 결과")
        print("   - 고차원 데이터에서 효과적")
        print("   - 모델에 독립적 (어떤 ML 모델과도 사용 가능)")
        print()
        print("⚠️  주의사항:")
        print("   - 특성 간 상관관계가 있으면 knockoff 생성이 어려움")
        print("   - 보수적인 방법 (적은 수의 특성 선택)")
        print("   - 충분한 샘플 크기 필요")
    
    def step2_create_knockoffs(self, X, method='equicorrelated'):
        """
        2단계: Knockoff 변수를 생성합니다.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            원본 특성 데이터
        method : str
            Knockoff 생성 방법 ('equicorrelated', 'sdp', 'gaussian')
        """
        print("\n🔄 2단계: Knockoff 변수 생성")
        print("=" * 40)
        
        self.X_original = X.copy()
        self.feature_names = X.columns.tolist()
        n_features = len(self.feature_names)
        
        print(f"📊 원본 데이터: {X.shape}")
        print(f"📊 특성 개수: {n_features}")
        print(f"📊 Knockoff 생성 방법: {method}")
        
        # 데이터 스케일링
        X_scaled = self.scaler.fit_transform(X)
        
        if method == 'equicorrelated':
            X_knockoffs = self._create_equicorrelated_knockoffs(X_scaled)
        elif method == 'gaussian':
            X_knockoffs = self._create_gaussian_knockoffs(X_scaled)
        elif method == 'permutation':
            X_knockoffs = self._create_permutation_knockoffs(X_scaled)
        else:
            raise ValueError(f"지원하지 않는 방법: {method}")
        
        # Knockoff 변수명 생성
        knockoff_names = [f"{name}_knockoff" for name in self.feature_names]
        
        # DataFrame으로 변환
        self.X_knockoffs = pd.DataFrame(
            X_knockoffs, 
            columns=knockoff_names, 
            index=X.index
        )
        
        print(f"✅ Knockoff 생성 완료: {self.X_knockoffs.shape}")
        
        # 상관관계 체크
        self._validate_knockoffs(X_scaled, X_knockoffs)
        
        return self.X_knockoffs
    
    def _create_equicorrelated_knockoffs(self, X):
        """
        Equicorrelated knockoffs를 생성합니다.
        가장 간단하고 일반적인 방법입니다.
        """
        print("   🔧 Equicorrelated knockoffs 생성 중...")
        
        n_samples, n_features = X.shape
        
        # 공분산 행렬 계산
        Sigma = np.cov(X.T)
        
        # Equicorrelated 파라미터 계산
        # s = min(1, lambda_min(2*Sigma))의 절반
        eigenvals = np.linalg.eigvals(2 * Sigma)
        s = min(1.0, np.min(eigenvals)) * 0.5
        
        print(f"      Equicorrelated 파라미터 s: {s:.4f}")
        
        # Knockoff 생성을 위한 변환 행렬
        try:
            Sigma_inv = np.linalg.inv(Sigma)
            G = np.linalg.cholesky(2 * s * Sigma - s**2 * np.eye(n_features))
            
            # 조건부 평균과 분산 계산
            mu_knockoff = X - s * X @ Sigma_inv
            
            # 노이즈 생성
            noise = np.random.multivariate_normal(
                np.zeros(n_features), 
                np.eye(n_features), 
                size=n_samples
            )
            
            # Knockoff 변수 생성
            X_knockoffs = mu_knockoff + noise @ G.T
            
        except np.linalg.LinAlgError:
            print("      ⚠️ 행렬 연산 오류 - Permutation 방법으로 대체")
            X_knockoffs = self._create_permutation_knockoffs(X)
        
        return X_knockoffs
    
    def _create_gaussian_knockoffs(self, X):
        """
        Gaussian knockoffs를 생성합니다.
        """
        print("   🔧 Gaussian knockoffs 생성 중...")
        
        n_samples, n_features = X.shape
        
        # 각 특성을 독립적으로 처리
        X_knockoffs = np.zeros_like(X)
        
        for j in range(n_features):
            # 다른 특성들로 현재 특성을 예측하는 선형 회귀
            X_others = np.delete(X, j, axis=1)
            
            if X_others.shape[1] > 0:
                # 최소제곱 해
                try:
                    beta = np.linalg.lstsq(X_others, X[:, j], rcond=None)[0]
                    residuals = X[:, j] - X_others @ beta
                    
                    # 잔차를 이용한 knockoff 생성
                    np.random.seed(self.random_state + j)
                    shuffled_residuals = np.random.permutation(residuals)
                    X_knockoffs[:, j] = X_others @ beta + shuffled_residuals
                    
                except np.linalg.LinAlgError:
                    # 문제가 있으면 단순 permutation
                    np.random.seed(self.random_state + j)
                    X_knockoffs[:, j] = np.random.permutation(X[:, j])
            else:
                np.random.seed(self.random_state + j)
                X_knockoffs[:, j] = np.random.permutation(X[:, j])
        
        return X_knockoffs
    
    def _create_permutation_knockoffs(self, X):
        """
        Permutation knockoffs를 생성합니다 (가장 단순한 방법).
        """
        print("   🔧 Permutation knockoffs 생성 중...")
        
        np.random.seed(self.random_state)
        X_knockoffs = np.zeros_like(X)
        
        for j in range(X.shape[1]):
            X_knockoffs[:, j] = np.random.permutation(X[:, j])
        
        return X_knockoffs
    
    def _validate_knockoffs(self, X_original, X_knockoffs):
        """
        생성된 knockoffs의 품질을 검증합니다.
        """
        print("\n   📊 Knockoff 품질 검증:")
        
        # 원본과 knockoff 간 상관관계 (낮아야 좋음)
        correlations = []
        for j in range(X_original.shape[1]):
            corr = np.corrcoef(X_original[:, j], X_knockoffs[:, j])[0, 1]
            correlations.append(abs(corr))
        
        mean_corr = np.mean(correlations)
        max_corr = np.max(correlations)
        
        print(f"      원본-Knockoff 평균 상관관계: {mean_corr:.4f}")
        print(f"      원본-Knockoff 최대 상관관계: {max_corr:.4f}")
        
        if mean_corr < 0.3:
            print("      ✅ 좋은 품질의 knockoffs")
        elif mean_corr < 0.5:
            print("      ⚠️ 보통 품질의 knockoffs")
        else:
            print("      ❌ 낮은 품질의 knockoffs - 결과 해석 주의")
    
    def step3_compute_knockoff_statistics(self, y, model_type='lasso'):
        """
        3단계: Knockoff 통계량을 계산합니다.
        
        Parameters:
        -----------
        y : pandas.Series
            타겟 변수
        model_type : str
            사용할 모델 ('lasso', 'logistic')
        """
        print("\n📊 3단계: Knockoff 통계량 계산")
        print("=" * 40)
        
        if self.X_original is None or self.X_knockoffs is None:
            print("❌ step2_create_knockoffs()를 먼저 실행하세요!")
            return None
        
        self.y = y
        
        # 원본과 knockoff 결합
        X_augmented = pd.concat([
            pd.DataFrame(self.scaler.transform(self.X_original), 
                        columns=self.feature_names, 
                        index=self.X_original.index),
            self.X_knockoffs
        ], axis=1)
        
        print(f"📊 결합된 데이터: {X_augmented.shape}")
        print(f"📊 모델 타입: {model_type}")
        
        # 모델 학습 및 중요도 계산
        if model_type == 'lasso':
            importance_scores = self._compute_lasso_importance(X_augmented, y)
        elif model_type == 'logistic':
            importance_scores = self._compute_logistic_importance(X_augmented, y)
        else:
            raise ValueError(f"지원하지 않는 모델: {model_type}")
        
        # Knockoff 통계량 계산
        self.knockoff_stats = self._compute_knockoff_stats(importance_scores)
        
        print(f"✅ Knockoff 통계량 계산 완료")
        print(f"   양수 통계량: {np.sum(self.knockoff_stats > 0)}개")
        print(f"   음수 통계량: {np.sum(self.knockoff_stats < 0)}개")
        print(f"   0 통계량: {np.sum(self.knockoff_stats == 0)}개")
        
        return self.knockoff_stats
    
    def _compute_lasso_importance(self, X_augmented, y):
        """
        Lasso 회귀를 사용하여 중요도를 계산합니다.
        """
        print("   🔧 Lasso 중요도 계산 중...")
        
        # 교차검증으로 최적 alpha 찾기
        lasso_cv = LassoCV(cv=5, random_state=self.random_state, max_iter=1000)
        lasso_cv.fit(X_augmented, y)
        
        # 계수의 절댓값이 중요도
        importance_scores = np.abs(lasso_cv.coef_)
        
        print(f"      최적 alpha: {lasso_cv.alpha_:.6f}")
        print(f"      선택된 특성: {np.sum(importance_scores > 1e-6)}개")
        
        return importance_scores
    
    def _compute_logistic_importance(self, X_augmented, y):
        """
        로지스틱 회귀를 사용하여 중요도를 계산합니다.
        """
        print("   🔧 로지스틱 회귀 중요도 계산 중...")
        
        # L1 정규화 로지스틱 회귀
        logistic = LogisticRegression(
            penalty='l1', 
            solver='liblinear',
            C=1.0,
            random_state=self.random_state,
            max_iter=1000
        )
        
        logistic.fit(X_augmented, y)
        
        # 다중 클래스의 경우 계수의 절댓값 합
        if len(logistic.coef_.shape) > 1:
            importance_scores = np.sum(np.abs(logistic.coef_), axis=0)
        else:
            importance_scores = np.abs(logistic.coef_[0])
        
        print(f"      선택된 특성: {np.sum(importance_scores > 1e-6)}개")
        
        return importance_scores
    
    def _compute_knockoff_stats(self, importance_scores):
        """
        Knockoff 통계량 W_j = |β_j| - |β̃_j|를 계산합니다.
        """
        n_features = len(self.feature_names)
        
        # 원본과 knockoff 중요도 분리
        original_importance = importance_scores[:n_features]
        knockoff_importance = importance_scores[n_features:]
        
        # Knockoff 통계량: W_j = |β_j| - |β̃_j|
        knockoff_stats = original_importance - knockoff_importance
        
        return knockoff_stats
    
    def step4_select_features(self, knockoff_plus=True):
        """
        4단계: FDR 제어 하에서 특성을 선택합니다.
        
        Parameters:
        -----------
        knockoff_plus : bool
            Knockoff+ 방법 사용 여부 (더 안전한 선택)
        """
        print("\n🎯 4단계: 특성 선택 (FDR 제어)")
        print("=" * 40)
        
        if self.knockoff_stats is None:
            print("❌ step3_compute_knockoff_statistics()를 먼저 실행하세요!")
            return None
        
        method_name = "Knockoff+" if knockoff_plus else "Knockoff"
        print(f"📊 선택 방법: {method_name}")
        print(f"📊 목표 FDR: {self.fdr_level}")
        
        # 임계값 계산
        if knockoff_plus:
            self.threshold = self._compute_knockoff_plus_threshold()
        else:
            self.threshold = self._compute_knockoff_threshold()
        
        # 특성 선택
        selected_mask = self.knockoff_stats >= self.threshold
        self.selected_features = [
            self.feature_names[i] for i in range(len(self.feature_names)) 
            if selected_mask[i]
        ]
        
        print(f"📊 계산된 임계값: {self.threshold:.4f}")
        print(f"📊 선택된 특성 수: {len(self.selected_features)}")
        print(f"📊 전체 특성 수: {len(self.feature_names)}")
        print(f"📊 선택 비율: {len(self.selected_features)/len(self.feature_names)*100:.1f}%")
        
        if len(self.selected_features) > 0:
            print(f"\n✅ 선택된 특성들:")
            for i, feature in enumerate(self.selected_features, 1):
                stat_value = self.knockoff_stats[self.feature_names.index(feature)]
                print(f"   {i:2d}. {feature:30} (W = {stat_value:7.4f})")
        else:
            print("\n⚠️ 선택된 특성이 없습니다. FDR 수준을 높이거나 데이터를 확인하세요.")
        
        return self.selected_features
    
    def _compute_knockoff_threshold(self):
        """
        기본 Knockoff 임계값을 계산합니다.
        """
        W = self.knockoff_stats
        
        # 양수 통계량들을 내림차순 정렬
        positive_stats = W[W > 0]
        if len(positive_stats) == 0:
            return float('inf')  # 선택되는 특성 없음
        
        positive_stats_sorted = np.sort(positive_stats)[::-1]
        
        for t in positive_stats_sorted:
            # FDR 추정
            false_discoveries = np.sum(W <= -t)
            discoveries = np.sum(W >= t)
            
            if discoveries > 0:
                fdr_estimate = false_discoveries / discoveries
                if fdr_estimate <= self.fdr_level:
                    return t
        
        return float('inf')
    
    def _compute_knockoff_plus_threshold(self):
        """
        Knockoff+ 임계값을 계산합니다 (더 보수적).
        """
        W = self.knockoff_stats
        
        positive_stats = W[W > 0]
        if len(positive_stats) == 0:
            return float('inf')
        
        positive_stats_sorted = np.sort(positive_stats)[::-1]
        
        for t in positive_stats_sorted:
            # Knockoff+에서는 분자에 1을 추가
            false_discoveries = np.sum(W <= -t) + 1
            discoveries = np.sum(W >= t)
            
            if discoveries > 0:
                fdr_estimate = false_discoveries / discoveries
                if fdr_estimate <= self.fdr_level:
                    return t
        
        return float('inf')
    
    def step5_visualize_results(self):
        """
        5단계: Knockoff 결과를 시각화합니다.
        """
        print("\n🎨 5단계: 결과 시각화")
        print("=" * 30)
        
        if self.knockoff_stats is None:
            print("❌ step3_compute_knockoff_statistics()를 먼저 실행하세요!")
            return
        
        # 서브플롯 설정
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Knockoff 통계량 분포
        self._plot_knockoff_statistics(axes[0, 0])
        
        # 2. 특성 선택 결과
        self._plot_feature_selection(axes[0, 1])
        
        # 3. 원본 vs Knockoff 중요도 비교
        self._plot_importance_comparison(axes[1, 0])
        
        # 4. FDR 분석
        self._plot_fdr_analysis(axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def _plot_knockoff_statistics(self, ax):
        """Knockoff 통계량 분포를 시각화합니다."""
        W = self.knockoff_stats
        
        ax.hist(W, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='W = 0')
        
        if self.threshold is not None and self.threshold != float('inf'):
            ax.axvline(x=self.threshold, color='green', linestyle='-', 
                      linewidth=2, label=f'임계값 = {self.threshold:.3f}')
        
        ax.set_xlabel('Knockoff 통계량 (W)')
        ax.set_ylabel('빈도')
        ax.set_title('Knockoff 통계량 분포')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_feature_selection(self, ax):
        """특성 선택 결과를 시각화합니다."""
        W = self.knockoff_stats
        colors = ['red' if w >= self.threshold else 'blue' for w in W]
        
        indices = range(len(W))
        ax.scatter(indices, W, c=colors, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        if self.threshold != float('inf'):
            ax.axhline(y=self.threshold, color='green', linestyle='-', 
                      linewidth=2, label=f'임계값 = {self.threshold:.3f}')
        
        ax.set_xlabel('특성 인덱스')
        ax.set_ylabel('Knockoff 통계량')
        ax.set_title('특성별 Knockoff 통계량')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_importance_comparison(self, ax):
        """원본과 Knockoff 중요도를 비교합니다."""
        # 이 부분은 중요도 점수가 저장되어 있어야 구현 가능
        ax.text(0.5, 0.5, '원본 vs Knockoff\n중요도 비교\n(구현 예정)', 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('원본 vs Knockoff 중요도')
    
    def _plot_fdr_analysis(self, ax):
        """FDR 분석 결과를 시각화합니다."""
        W = self.knockoff_stats
        thresholds = np.linspace(0, np.max(W), 100)
        fdr_estimates = []
        
        for t in thresholds:
            false_discoveries = np.sum(W <= -t) + 1
            discoveries = np.sum(W >= t)
            fdr_est = false_discoveries / max(discoveries, 1)
            fdr_estimates.append(fdr_est)
        
        ax.plot(thresholds, fdr_estimates, 'b-', linewidth=2)
        ax.axhline(y=self.fdr_level, color='red', linestyle='--', 
                  label=f'목표 FDR = {self.fdr_level}')
        
        if self.threshold != float('inf'):
            ax.axvline(x=self.threshold, color='green', linestyle='-',
                      linewidth=2, label=f'선택된 임계값 = {self.threshold:.3f}')
        
        ax.set_xlabel('임계값')
        ax.set_ylabel('추정 FDR')
        ax.set_title('FDR vs 임계값')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def get_summary(self):
        """분석 결과를 요약합니다."""
        print("\n📋 Knockoff Selection 결과 요약")
        print("=" * 50)
        
        if self.X_original is not None:
            print(f"✅ 원본 데이터: {self.X_original.shape}")
        
        if self.X_knockoffs is not None:
            print(f"✅ Knockoff 생성: {self.X_knockoffs.shape}")
        
        if self.knockoff_stats is not None:
            print(f"✅ Knockoff 통계량: {len(self.knockoff_stats)}개")
            print(f"   양수: {np.sum(self.knockoff_stats > 0)}개")
            print(f"   음수: {np.sum(self.knockoff_stats < 0)}개")
        
        if self.selected_features is not None:
            print(f"✅ 선택된 특성: {len(self.selected_features)}개")
            print(f"   선택 비율: {len(self.selected_features)/len(self.feature_names)*100:.1f}%")
            print(f"   FDR 수준: {self.fdr_level}")
            
            if len(self.selected_features) > 0:
                print(f"\n🎯 선택된 특성 목록:")
                for feature in self.selected_features:
                    print(f"   • {feature}")

# 사용 예시
def run_knockoff_analysis(X, y, fdr_level=0.1, knockoff_method='equicorrelated'):
    """
    전체 Knockoff Selection 분석을 실행합니다.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        특성 데이터
    y : pandas.Series  
        타겟 데이터
    fdr_level : float
        원하는 FDR 수준
    knockoff_method : str
        Knockoff 생성 방법
    
    Returns:
    --------
    KnockoffSelection
        분석이 완료된 객체
    """
    
    print("🚀 Knockoff Selection 분석 시작")
    print("=" * 50)
    
    # 1. 객체 생성
    ko = KnockoffSelection(fdr_level=fdr_level)
    
    # 2. 개념 설명
    ko.step1_explain_knockoffs()
    
    # 3. Knockoff 생성
    ko.step2_create_knockoffs(X, method=knockoff_method)
    
    # 4. 통계량 계산
    ko.step3_compute_knockoff_statistics(y, model_type='logistic')
    
    # 5. 특성 선택
    selected_features = ko.step4_select_features(knockoff_plus=True)
    
    # 6. 시각화
    ko.step5_visualize_results()
    
    # 7. 요약
    ko.get_summary()
    
    return ko

if __name__ == "__main__":
    print("💡 Knockoff Selection 사용법:")
    print("=" * 40)
    print("# 1. 기본 실행")
    print("ko = run_knockoff_analysis(X, y, fdr_level=0.1)")
    print()
    print("# 2. 단계별 실행") 
    print("ko = KnockoffSelection(fdr_level=0.1)")
    print("ko.step1_explain_knockoffs()")
    print("ko.step2_create_knockoffs(X, method='equicorrelated')")
    print("ko.step3_compute_knockoff_statistics(y, model_type='logistic')")
    print("ko.step4_select_features(knockoff_plus=True)")
    print("ko.step5_visualize_results()")
    print("ko.get_summary()")
    print()
    print("# 3. 다른 방법들")
    print("# - Knockoff 생성: 'equicorrelated', 'gaussian', 'permutation'") 
    print("# - 모델 타입: 'lasso', 'logistic'")
    print("# - FDR 수준: 0.05, 0.1, 0.2 등")
    print("# - Knockoff vs Knockoff+ 방법")
    print()
    print("🎯 주요 특징:")
    print("  - 통계적으로 엄격한 특성 선택")
    print("  - False Discovery Rate 제어")
    print("  - 고차원 데이터에 적합")
    print("  - 모델에 독립적")
    print("  - 재현 가능한 결과")

# 추가 유틸리티 함수들
class KnockoffUtils:
    """Knockoff Selection을 위한 유틸리티 함수들"""
    
    @staticmethod
    def compare_methods(X, y, fdr_levels=[0.05, 0.1, 0.2], 
                       knockoff_methods=['equicorrelated', 'gaussian', 'permutation']):
        """
        다양한 FDR 수준과 Knockoff 방법을 비교합니다.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            특성 데이터
        y : pandas.Series
            타겟 데이터
        fdr_levels : list
            테스트할 FDR 수준들
        knockoff_methods : list
            테스트할 Knockoff 생성 방법들
        
        Returns:
        --------
        pandas.DataFrame
            비교 결과 테이블
        """
        
        print("🔍 Knockoff 방법들 비교 분석")
        print("=" * 40)
        
        results = []
        
        for fdr in fdr_levels:
            for method in knockoff_methods:
                try:
                    print(f"\n📊 FDR={fdr}, Method={method}")
                    
                    ko = KnockoffSelection(fdr_level=fdr)
                    ko.step2_create_knockoffs(X, method=method)
                    ko.step3_compute_knockoff_statistics(y, model_type='logistic')
                    selected = ko.step4_select_features(knockoff_plus=True)
                    
                    results.append({
                        'FDR_Level': fdr,
                        'Method': method,
                        'Selected_Features': len(selected) if selected else 0,
                        'Selection_Rate': (len(selected) / len(X.columns) * 100) if selected else 0,
                        'Threshold': ko.threshold if ko.threshold != float('inf') else 'inf',
                        'Positive_Stats': np.sum(ko.knockoff_stats > 0),
                        'Negative_Stats': np.sum(ko.knockoff_stats < 0)
                    })
                    
                except Exception as e:
                    print(f"   ❌ 오류: {str(e)[:50]}...")
                    results.append({
                        'FDR_Level': fdr,
                        'Method': method,
                        'Selected_Features': 0,
                        'Selection_Rate': 0,
                        'Threshold': 'error',
                        'Positive_Stats': 0,
                        'Negative_Stats': 0
                    })
        
        comparison_df = pd.DataFrame(results)
        
        print("\n📋 비교 결과:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    @staticmethod
    def evaluate_knockoff_quality(X_original, X_knockoffs):
        """
        생성된 Knockoff의 품질을 자세히 평가합니다.
        
        Parameters:
        -----------
        X_original : numpy.ndarray
            원본 특성 데이터
        X_knockoffs : numpy.ndarray
            Knockoff 특성 데이터
        
        Returns:
        --------
        dict
            품질 평가 결과
        """
        
        n_features = X_original.shape[1]
        
        # 1. 개별 특성별 상관관계
        individual_corrs = []
        for j in range(n_features):
            corr = np.corrcoef(X_original[:, j], X_knockoffs[:, j])[0, 1]
            individual_corrs.append(abs(corr))
        
        # 2. 교차 상관관계 (원본 i와 knockoff j, i≠j)
        cross_corrs = []
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    corr = np.corrcoef(X_original[:, i], X_knockoffs[:, j])[0, 1]
                    cross_corrs.append(abs(corr))
        
        # 3. 분포 유사성 (Kolmogorov-Smirnov 테스트)
        ks_statistics = []
        for j in range(n_features):
            ks_stat, _ = stats.ks_2samp(X_original[:, j], X_knockoffs[:, j])
            ks_statistics.append(ks_stat)
        
        quality_metrics = {
            'mean_individual_corr': np.mean(individual_corrs),
            'max_individual_corr': np.max(individual_corrs),
            'mean_cross_corr': np.mean(cross_corrs),
            'max_cross_corr': np.max(cross_corrs),
            'mean_ks_statistic': np.mean(ks_statistics),
            'max_ks_statistic': np.max(ks_statistics),
            'individual_corrs': individual_corrs,
            'quality_score': None
        }
        
        # 종합 품질 점수 (0-100, 높을수록 좋음)
        # 낮은 상관관계와 유사한 분포가 좋음
        corr_score = max(0, 100 * (1 - quality_metrics['mean_individual_corr']))
        dist_score = max(0, 100 * (1 - quality_metrics['mean_ks_statistic']))
        quality_metrics['quality_score'] = (corr_score + dist_score) / 2
        
        return quality_metrics
    
    @staticmethod
    def plot_knockoff_comparison(ko_results_list, method_names):
        """
        여러 Knockoff 방법의 결과를 비교 시각화합니다.
        
        Parameters:
        -----------
        ko_results_list : list
            KnockoffSelection 객체들의 리스트
        method_names : list
            각 방법의 이름
        """
        
        n_methods = len(ko_results_list)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_methods))
        
        # 1. Knockoff 통계량 분포 비교
        ax = axes[0, 0]
        for i, (ko, name) in enumerate(zip(ko_results_list, method_names)):
            ax.hist(ko.knockoff_stats, bins=20, alpha=0.6, 
                   label=name, color=colors[i])
        ax.set_xlabel('Knockoff 통계량')
        ax.set_ylabel('빈도')
        ax.set_title('Knockoff 통계량 분포 비교')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 선택된 특성 수 비교
        ax = axes[0, 1]
        selected_counts = [len(ko.selected_features) if ko.selected_features else 0 
                          for ko in ko_results_list]
        bars = ax.bar(method_names, selected_counts, color=colors)
        ax.set_ylabel('선택된 특성 수')
        ax.set_title('방법별 선택된 특성 수')
        ax.grid(True, alpha=0.3)
        
        # 막대 위에 숫자 표시
        for bar, count in zip(bars, selected_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom')
        
        # 3. 임계값 비교
        ax = axes[1, 0]
        thresholds = [ko.threshold if ko.threshold != float('inf') else 0 
                     for ko in ko_results_list]
        bars = ax.bar(method_names, thresholds, color=colors)
        ax.set_ylabel('임계값')
        ax.set_title('방법별 선택 임계값')
        ax.grid(True, alpha=0.3)
        
        # 4. 양수/음수 통계량 비교
        ax = axes[1, 1]
        positive_counts = [np.sum(ko.knockoff_stats > 0) for ko in ko_results_list]
        negative_counts = [np.sum(ko.knockoff_stats < 0) for ko in ko_results_list]
        
        x = np.arange(len(method_names))
        width = 0.35
        
        ax.bar(x - width/2, positive_counts, width, label='양수', alpha=0.8)
        ax.bar(x + width/2, negative_counts, width, label='음수', alpha=0.8)
        
        ax.set_xlabel('방법')
        ax.set_ylabel('통계량 개수')
        ax.set_title('양수/음수 통계량 개수')
        ax.set_xticks(x)
        ax.set_xticklabels(method_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_feature_importance_report(ko, feature_descriptions=None):
        """
        선택된 특성들에 대한 상세 리포트를 생성합니다.
        
        Parameters:
        -----------
        ko : KnockoffSelection
            분석이 완료된 Knockoff 객체
        feature_descriptions : dict
            특성명 -> 설명 매핑 딕셔너리
        
        Returns:
        --------
        pandas.DataFrame
            특성 중요도 리포트
        """
        
        if ko.selected_features is None:
            print("❌ 특성 선택이 완료되지 않았습니다.")
            return None
        
        report_data = []
        
        for feature in ko.selected_features:
            idx = ko.feature_names.index(feature)
            knockoff_stat = ko.knockoff_stats[idx]
            
            # 순위 계산 (높은 통계량부터)
            rank = np.sum(ko.knockoff_stats > knockoff_stat) + 1
            
            # 설명 추가
            description = ""
            if feature_descriptions and feature in feature_descriptions:
                description = feature_descriptions[feature]
            
            report_data.append({
                'Feature': feature,
                'Knockoff_Statistic': knockoff_stat,
                'Rank': rank,
                'Description': description,
                'Significance': 'High' if knockoff_stat > ko.threshold * 2 else 'Medium'
            })
        
        # 통계량 기준으로 정렬
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('Knockoff_Statistic', ascending=False)
        report_df.reset_index(drop=True, inplace=True)
        
        print(f"\n📊 선택된 특성 상세 리포트 (FDR ≤ {ko.fdr_level})")
        print("=" * 80)
        print(report_df.to_string(index=False))
        
        return report_df

# 실제 사용 예시를 위한 헬퍼 함수
def demo_knockoff_with_cancer_data(X, y, target_col='암종'):
    """
    암종 데이터를 사용한 Knockoff Selection 데모
    
    Parameters:
    -----------
    X : pandas.DataFrame
        특성 데이터 (수치형 + 카테고리형)
    y : pandas.Series
        암종 타겟 데이터
    target_col : str
        타겟 컬럼명
    """
    
    print("🏥 암종 데이터 Knockoff Selection 데모")
    print("=" * 50)
    
    # 데이터 정보 출력
    print(f"📊 데이터 크기: {X.shape}")
    print(f"📊 타겟 분포:")
    print(y.value_counts())
    print()
    
    # 1. 기본 분석 (FDR = 0.1)
    print("🎯 기본 분석 (FDR = 0.1)")
    ko_basic = run_knockoff_analysis(X, y, fdr_level=0.1, 
                                   knockoff_method='equicorrelated')
    
    # 2. 엄격한 분석 (FDR = 0.05)
    print("\n🎯 엄격한 분석 (FDR = 0.05)")
    ko_strict = KnockoffSelection(fdr_level=0.05)
    ko_strict.step2_create_knockoffs(X, method='equicorrelated')
    ko_strict.step3_compute_knockoff_statistics(y, model_type='logistic')
    ko_strict.step4_select_features(knockoff_plus=True)
    
    # 3. 완화된 분석 (FDR = 0.2)
    print("\n🎯 완화된 분석 (FDR = 0.2)")
    ko_relaxed = KnockoffSelection(fdr_level=0.2)
    ko_relaxed.step2_create_knockoffs(X, method='equicorrelated')
    ko_relaxed.step3_compute_knockoff_statistics(y, model_type='logistic')
    ko_relaxed.step4_select_features(knockoff_plus=True)
    
    # 4. 결과 비교
    print("\n📊 FDR 수준별 결과 비교")
    print("=" * 40)
    
    comparison_data = [
        {
            'FDR_Level': 0.05,
            'Selected_Features': len(ko_strict.selected_features) if ko_strict.selected_features else 0,
            'Threshold': ko_strict.threshold if ko_strict.threshold != float('inf') else 'inf'
        },
        {
            'FDR_Level': 0.10,
            'Selected_Features': len(ko_basic.selected_features) if ko_basic.selected_features else 0,
            'Threshold': ko_basic.threshold if ko_basic.threshold != float('inf') else 'inf'
        },
        {
            'FDR_Level': 0.20,
            'Selected_Features': len(ko_relaxed.selected_features) if ko_relaxed.selected_features else 0,
            'Threshold': ko_relaxed.threshold if ko_relaxed.threshold != float('inf') else 'inf'
        }
    ]
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # 5. 방법 비교 시각화
    KnockoffUtils.plot_knockoff_comparison(
        [ko_strict, ko_basic, ko_relaxed],
        ['FDR=0.05', 'FDR=0.10', 'FDR=0.20']
    )
    
    return ko_basic, ko_strict, ko_relaxed