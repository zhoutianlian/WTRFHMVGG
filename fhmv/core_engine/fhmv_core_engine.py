# fhmv/core_engine/fhmv_core_engine.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.special import gammaln, digamma # psi is equivalent to digamma
from scipy.linalg import det, inv, pinv # For matrix determinant, inverse, pseudo-inverse
from scipy.optimize import brentq, minimize 
from itertools import product
import traceback # For more detailed error printouts

class FHMVCoreEngine:
    """
    Factor-based Hidden Markov Model Variant (FHMV) Core Engine.

    Implements the FHMV model with Student's t-Mixture Model (StMM) emissions.
    Handles model training via the Expectation-Maximization (EM) algorithm,
    including Expectation (E) and Maximization (M) steps. The M-step includes
    updates for StMM parameters (weights, means, covariances, Degrees of Freedom)
    and FHMV-specific parameters (transition parameters for persistence/jump factors,
    and magnitude/scaling parameters for persistence, jump, leverage, and overall variance).
    Also provides inference methods like Viterbi for the most likely state sequence
    and calculation of state probabilities.

    The FHMV structure involves combined states derived from underlying factorial
    components (Persistence, Jump, Leverage).
    """

    def __init__(self, config: dict):
        """
        Initializes the FHMVCoreEngine with parameters from a configuration dictionary.
        """
        self.num_fhmv_regimes = config.get("num_fhmv_regimes", 6)
        self.num_persistence_chains = config.get("num_persistence_chains", 2)
        self.num_jump_states = config.get("num_jump_states", 2)      
        self.num_leverage_states = config.get("num_leverage_states", 2)   
        self.num_stmm_mix_components = config.get("num_stmm_mix_components", 2) # K_mix
        self.num_features = config.get("num_features", 10) # D
        
        self.em_max_iterations = config.get("em_max_iterations", 100)
        self.em_tolerance = config.get("em_tolerance", 1e-4)
        self.dof_min = config.get("dof_min", 2.01) 
        self.dof_max = config.get("dof_max", 200.0)
        
        self.combined_to_final_regime_mapping = config.get("combined_to_final_regime_mapping", None)
        if self.combined_to_final_regime_mapping:
            if not all(isinstance(k, int) and 0 <= k < ( (2**self.num_persistence_chains) * self.num_jump_states * self.num_leverage_states) for k in self.combined_to_final_regime_mapping.keys()):
                raise ValueError("Invalid keys in combined_to_final_regime_mapping.")
            if not all(isinstance(v, int) and 0 <= v < self.num_fhmv_regimes for v in self.combined_to_final_regime_mapping.values()):
                 raise ValueError(f"Invalid values in combined_to_final_regime_mapping (should be 0 to {self.num_fhmv_regimes-1}).")

        # FHMV Component Transition/Dynamic Parameters (some fixed by config, some learned)
        self.leverage_trend_threshold_pos = config.get("leverage_trend_threshold_pos", 0.1) 
        self.leverage_trend_threshold_neg = config.get("leverage_trend_threshold_neg", -0.1)
        self.leverage_prob_stay_favored = config.get("leverage_prob_stay_favored", 0.9) 
        self.leverage_prob_stay_unfavored = config.get("leverage_prob_stay_unfavored", 0.7) 
        self.q_jump_revert_init = config.get("q_jump_revert_init",0.9) # Initial value for q_jump_revert

        self.min_prob_transition_param = config.get("min_prob_transition_param", 0.01)
        self.max_prob_transition_param = config.get("max_prob_transition_param", 0.99)
        
        # Bounds for FHMV magnitude parameters optimization
        self.fhmv_magnitude_param_bounds = {
            'c1_persistence': tuple(config.get("c1_bounds", [1.001, 3.0])), 
            'theta_c_persistence': tuple(config.get("theta_c_bounds", [0.1, 0.99])),
            'm1_jump': tuple(config.get("m1_bounds", [1.001, 5.0])), 
            'theta_m_jump': tuple(config.get("theta_m_bounds", [0.1, 0.99])),
            'l1_leverage': tuple(config.get("l1_bounds", [1.001, 3.0])), 
            'theta_l_leverage': tuple(config.get("theta_l_bounds", [0.1, 0.99])),
            'sigma_sq_overall': tuple(config.get("sigma_sq_bounds", [1e-4, 10.0]))
        }

        # Core Model Parameters (to be learned or initialized)
        self.fhmv_params_ = None # Stores p_persistence, c1, q_jump, m1, l1, sigma_sq etc.
        self.stmm_emission_params_ = None # Stores mu, w, nu, and SIGMA_BASE for each state's StMM
        self.initial_state_probs_ = None  # For combined FHMV states

        self.num_combined_fhmv_states = (2**self.num_persistence_chains) * \
                                        self.num_jump_states * \
                                        self.num_leverage_states
        
        # Helper structures for state mapping
        self._persistence_state_tuples = list(product([0, 1], repeat=self.num_persistence_chains))
        self._jump_state_tuples = list(range(self.num_jump_states))
        self._leverage_state_tuples = list(range(self.num_leverage_states))
        
        self._idx_to_comp_states = {} 
        self._comp_states_to_idx = {} 
        idx = 0
        for p_state_tuple_val in self._persistence_state_tuples:
            for m_state_val in self._jump_state_tuples:
                for l_state_val in self._leverage_state_tuples:
                    full_state_tuple = (p_state_tuple_val, m_state_val, l_state_val)
                    self._idx_to_comp_states[idx] = full_state_tuple
                    self._comp_states_to_idx[full_state_tuple] = idx
                    idx += 1
        if idx != self.num_combined_fhmv_states: 
            raise RuntimeError("FHMV state mapping generation failed during initialization.")
            
        print(f"FHMVCoreEngine Initialized with config, {self.num_combined_fhmv_states} combined FHMV states.")

    def _initialize_fhmv_component_parameters(self):
        """Initializes FHMV-specific factor parameters (transition and magnitude)."""
        # Initial guesses for transition parameters
        p_persistence = np.full(self.num_persistence_chains, 
                                np.clip(0.99, self.min_prob_transition_param, self.max_prob_transition_param))
        q_jump = np.clip(0.01, self.min_prob_transition_param, self.max_prob_transition_param) # P(S1|S0) for jump
        q_jump_revert = np.clip(self.q_jump_revert_init, self.min_prob_transition_param, self.max_prob_transition_param) # P(S0|S1) for jump

        # Initial guesses for magnitude parameters (often starting at lower bound or mean)
        c1_persistence = self.fhmv_magnitude_param_bounds['c1_persistence'][0] 
        theta_c_persistence = np.mean(self.fhmv_magnitude_param_bounds['theta_c_persistence'])
        m1_jump = self.fhmv_magnitude_param_bounds['m1_jump'][0]
        theta_m_jump = np.mean(self.fhmv_magnitude_param_bounds['theta_m_jump'])
        l1_leverage = self.fhmv_magnitude_param_bounds['l1_leverage'][0] 
        theta_l_leverage = np.mean(self.fhmv_magnitude_param_bounds['theta_l_leverage'])
        sigma_sq_overall = np.mean(self.fhmv_magnitude_param_bounds['sigma_sq_overall'])
        if sigma_sq_overall <=0: sigma_sq_overall = self.fhmv_magnitude_param_bounds['sigma_sq_overall'][0]


        self.fhmv_params_ = {
            'p_persistence': p_persistence, 
            'c1_persistence': c1_persistence, 'theta_c_persistence': theta_c_persistence, 
            'q_jump': q_jump, 'q_jump_revert': q_jump_revert, 
            'm1_jump': m1_jump, 'theta_m_jump': theta_m_jump,
            'l1_leverage': l1_leverage, 'theta_l_leverage': theta_l_leverage, 
            'sigma_sq_overall': sigma_sq_overall
        }
        # print("FHMV component parameters structure initialized.")

    def _initialize_stmm_emission_parameters(self, training_features: pd.DataFrame):
        """Initializes StMM emission parameters (means, base_covariances, weights, DoF)."""
        # (Code from previous step, ensure 'covariance' stores SIGMA_BASE)
        if training_features.isnull().values.any():
            cleaned_features = training_features.dropna()
            if cleaned_features.empty: raise ValueError("Training features all NaN after drop.")
        else: cleaned_features = training_features.copy() # Use copy to avoid modifying original
        
        if cleaned_features.shape[0] < self.num_stmm_mix_components:
             raise ValueError(f"Num samples ({cleaned_features.shape[0]}) < STMM mix components ({self.num_stmm_mix_components}).")

        kmeans = KMeans(n_clusters=self.num_stmm_mix_components, random_state=42, n_init='auto')
        try:
            kmeans.fit(cleaned_features.values)
        except Exception as e:
            raise RuntimeError(f"KMeans fitting failed during StMM initialization: {e}. Check input features.")
            
        initial_means_from_kmeans = kmeans.cluster_centers_
        
        global_feature_variances = np.maximum(cleaned_features.var(ddof=1).values, 1e-6) # Ensure positive
        
        initial_mix_weights = np.full(self.num_stmm_mix_components, 1.0 / self.num_stmm_mix_components)
        initial_dof = np.clip(5.0, self.dof_min, self.dof_max)
        
        self.stmm_emission_params_ = []
        for _ in range(self.num_combined_fhmv_states):
            state_emission_params = []
            for k_idx in range(self.num_stmm_mix_components):
                sigma_base_init = np.diag(global_feature_variances) 
                state_emission_params.append({
                    'weight': initial_mix_weights[k_idx],
                    'mean': initial_means_from_kmeans[k_idx, :].copy(),
                    'covariance': sigma_base_init.copy(), # This is Sigma_jk_BASE
                    'dof': initial_dof})
            self.stmm_emission_params_.append(state_emission_params)
        
        # sigma_sq_overall is initialized in _initialize_fhmv_component_parameters
        # print("StMM emission parameters (base) initialized.")


    def _initialize_state_parameters(self, num_states: int):
        """Initializes initial state probabilities."""
        self.initial_state_probs_ = np.full(num_states, 1.0 / num_states)
        # print("Initial state probabilities initialized.")

    def _initialize_all_parameters(self, training_features: pd.DataFrame):
        """Helper to call all parameter initializers."""
        self._initialize_fhmv_component_parameters()
        self._initialize_stmm_emission_parameters(training_features) # Needs data
        self._initialize_state_parameters(self.num_combined_fhmv_states)
        print("All FHMV Core Engine parameters initialized.")

    # --- Helper methods for FHMV factor calculation and PDF ---
    def _calculate_fhmv_factors_for_state(self, combined_state_idx: int, fhmv_mag_params_dict: dict) -> tuple[float, float, float]:
        """Calculates C_factor, M_factor, L_factor for a given combined_state_idx."""
        comp_states = self._idx_to_comp_states[combined_state_idx]
        p_states_tuple, m_state, l_state = comp_states
        
        c_factor = 1.0
        c1 = fhmv_mag_params_dict['c1_persistence']
        theta_c = fhmv_mag_params_dict['theta_c_persistence']
        for n in range(self.num_persistence_chains):
            if p_states_tuple[n] == 1: # Assuming state 1 is "ON"
                c_n = 1.0 + (theta_c**n) * (c1 - 1.0) 
                c_factor *= c_n
        
        m_factor = 1.0
        if self.num_jump_states > 1 and m_state > 0: # Assuming state 0 is "No Jump"
            m_factor = fhmv_mag_params_dict['m1_jump']
            
        l_factor = 1.0
        if self.num_leverage_states > 1 and l_state > 0: # Assuming state 0 is "Baseline Leverage"
            l_factor = fhmv_mag_params_dict['l1_leverage'] # l1_leverage is the factor itself
        
        return c_factor, m_factor, l_factor

    def _get_effective_covariance(self, combined_state_idx: int, component_k_idx: int, 
                                  fhmv_mag_params_dict: dict) -> np.ndarray:
        """ 
        Calculates Sigma_jk_EFFECTIVE based on Sigma_jk_BASE and FHMV factors.
        Enhanced with numerical stability safeguards.
        """
        C_j, M_j, L_j = self._calculate_fhmv_factors_for_state(combined_state_idx, fhmv_mag_params_dict)
        sigma_sq = fhmv_mag_params_dict['sigma_sq_overall']
        
        # Calculate factors in log space to prevent overflow/underflow
        log_factors = np.log(np.abs(sigma_sq) + 1e-30) + \
                     np.log(np.abs(C_j) + 1e-30) + \
                     np.log(np.abs(M_j) + 1e-30) + \
                     np.log(np.abs(L_j) + 1e-30)
        
        # Clamp to reasonable range to prevent extreme scaling
        log_factors = np.clip(log_factors, -20, 10)  # e^-20 to e^10 range
        effective_variance_scale = np.exp(log_factors)
        
        # Ensure minimum scale based on base covariance magnitude
        sigma_base_jk = self.stmm_emission_params_[combined_state_idx][component_k_idx]['covariance']
        min_scale = 1e-12 * np.trace(sigma_base_jk) / self.num_features
        effective_variance_scale = np.maximum(effective_variance_scale, min_scale)
            
        sigma_effective_jk = effective_variance_scale * sigma_base_jk
        
        # Enhanced positive definiteness ensuring
        min_eigenval = 1e-9 * np.trace(sigma_base_jk) / self.num_features
        sigma_effective_jk = sigma_effective_jk + np.eye(self.num_features) * min_eigenval
        
        return sigma_effective_jk

    def _multivariate_student_t_pdf(self, y_t: np.ndarray, mean: np.ndarray,
                                    covariance_effective: np.ndarray, dof: float, # Takes effective covariance
                                    log_pdf: bool = False) -> float:
        """
        Calculates PDF for multivariate Student's t-distribution using effective covariance.
        Enhanced with numerical stability and robust error handling.
        """
        D = self.num_features
        
        # Enhanced numerical stability for matrix operations
        try:
            # Use Cholesky decomposition for better numerical stability
            L = np.linalg.cholesky(covariance_effective)
            log_det_cov_eff = 2.0 * np.sum(np.log(np.diag(L)))
            
            # Solve using forward/back substitution for numerical stability
            diff = y_t - mean
            z = np.linalg.solve(L, diff)
            delta_sq = np.dot(z, z)
            
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse with regularization
            try:
                # Add regularization to ensure positive definiteness
                reg_cov = covariance_effective + np.eye(D) * 1e-6 * np.trace(covariance_effective) / D
                cov_eff_inv = pinv(reg_cov)
                sign, log_det_cov_eff = np.linalg.slogdet(reg_cov)
                if sign <= 0: 
                    return -np.inf if log_pdf else 0.0
                
                diff = y_t - mean
                delta_sq = diff.T @ cov_eff_inv @ diff
                
            except:
                return -np.inf if log_pdf else 0.0

        # Clamp delta_sq to reasonable range
        delta_sq = np.maximum(delta_sq, 0.0)  # Ensure non-negative
        delta_sq = np.minimum(delta_sq, 1e10)  # Prevent extreme values
        
        # Clamp dof to reasonable range
        dof_clamped = np.clip(dof, self.dof_min, self.dof_max)

        # Calculate log PDF components with enhanced numerical stability
        try:
            log_c = gammaln((dof_clamped + D) / 2.0) - gammaln(dof_clamped / 2.0) - \
                    (D / 2.0) * np.log(dof_clamped * np.pi) - 0.5 * log_det_cov_eff
            
            # Use log1p for better numerical stability when delta_sq/dof is small
            if delta_sq / dof_clamped < 0.1:
                log_p_data_term = -((dof_clamped + D) / 2.0) * np.log1p(delta_sq / dof_clamped)
            else:
                term_in_log = 1.0 + delta_sq / dof_clamped
                if term_in_log <= 1e-15:
                    log_p_data_term = -np.inf
                else:
                    log_p_data_term = -((dof_clamped + D) / 2.0) * np.log(term_in_log)

            log_p = log_c + log_p_data_term
            
            # Clamp final result to prevent extreme values
            log_p = np.clip(log_p, -1e10, 100)
            
        except (OverflowError, ZeroDivisionError, ValueError):
            return -np.inf if log_pdf else 0.0

        return log_p if log_pdf else np.exp(np.clip(log_p, -700, 700))

    # --- Transition Matrix Helpers ---
    def _get_persistence_transition_prob(self, prev_p_states: tuple, next_p_states: tuple) -> float:
        # (Code from previous step)
        if len(prev_p_states) != self.num_persistence_chains or len(next_p_states) != self.num_persistence_chains: raise ValueError("P-state length mismatch.")
        log_total_prob = 0.0; p_persistence_arr = self.fhmv_params_['p_persistence']
        for n in range(self.num_persistence_chains):
            p_n = p_persistence_arr[n]
            if prev_p_states[n] == next_p_states[n]: log_total_prob += np.log(p_n + 1e-30)
            else: log_total_prob += np.log(1.0 - p_n + 1e-30)
        return np.exp(log_total_prob)

    def _get_jump_transition_prob(self, prev_m_state: int, next_m_state: int) -> float:
        # (Code from previous step, uses self.fhmv_params_['q_jump_revert'])
        q_jump = self.fhmv_params_['q_jump'] 
        q_revert = self.fhmv_params_['q_jump_revert']
        if self.num_jump_states == 1: return 1.0 if prev_m_state == next_m_state else 0.0
        if self.num_jump_states == 2:
            if prev_m_state == 0: return (1.0 - q_jump) if next_m_state == 0 else q_jump
            elif prev_m_state == 1: return q_revert if next_m_state == 0 else (1.0 - q_revert)
            else: raise ValueError(f"Invalid previous jump state: {prev_m_state}")
        else: 
            if prev_m_state == next_m_state: return 0.9 # Placeholder
            else: return 0.1 / (self.num_jump_states - 1) if self.num_jump_states > 1 else 0.0

    def _get_leverage_transition_prob(self, prev_l_state: int, next_l_state: int, current_trend_feature: float = None) -> float:
        # (Code from previous step)
        if self.num_leverage_states == 1: return 1.0 if prev_l_state == next_l_state else 0.0
        if self.num_leverage_states == 2:
            prob_to_state0, prob_to_state1 = 0.5, 0.5 
            if current_trend_feature is not None:
                favored_state = -1 # Neutral
                if current_trend_feature > self.leverage_trend_threshold_pos: favored_state = 0 # State 0 favored
                elif current_trend_feature < self.leverage_trend_threshold_neg: favored_state = 1 # State 1 favored
                if prev_l_state == 0: 
                    if favored_state == 0: prob_to_state0 = self.leverage_prob_stay_favored; prob_to_state1 = 1.0 - prob_to_state0
                    elif favored_state == 1: prob_to_state1 = self.leverage_prob_stay_favored; prob_to_state0 = 1.0 - prob_to_state1
                    else: prob_to_state0 = self.leverage_prob_stay_unfavored; prob_to_state1 = 1.0 - prob_to_state0
                elif prev_l_state == 1: 
                    if favored_state == 1: prob_to_state1 = self.leverage_prob_stay_favored; prob_to_state0 = 1.0 - prob_to_state1
                    elif favored_state == 0: prob_to_state0 = self.leverage_prob_stay_favored; prob_to_state1 = 1.0 - prob_to_state0
                    else: prob_to_state1 = self.leverage_prob_stay_unfavored; prob_to_state0 = 1.0 - prob_to_state0
                else: raise ValueError(f"Invalid prev_l_state: {prev_l_state}")
            return prob_to_state0 if next_l_state == 0 else prob_to_state1
        else: # Placeholder
            if prev_l_state == next_l_state: return 0.9
            else: return 0.1 / (self.num_leverage_states - 1) if self.num_leverage_states > 1 else 0.0
            
    def _get_combined_transition_matrix_A_t(self, t: int, features_df: pd.DataFrame) -> np.ndarray:
        # (Code from previous step, uses features_df.iloc[t]['T'])
        A_t = np.zeros((self.num_combined_fhmv_states, self.num_combined_fhmv_states))
        current_trend_for_leverage = None
        if features_df is not None and 'T' in features_df.columns and t < len(features_df):
            current_trend_for_leverage = features_df.iloc[t]['T']
        for i_idx in range(self.num_combined_fhmv_states):
            prev_comp_states = self._idx_to_comp_states[i_idx]
            prev_p_states, prev_m_state, prev_l_state = prev_comp_states
            for j_idx in range(self.num_combined_fhmv_states):
                next_comp_states = self._idx_to_comp_states[j_idx]
                next_p_states, next_m_state, next_l_state = next_comp_states
                p_trans_c = self._get_persistence_transition_prob(prev_p_states, next_p_states)
                p_trans_m = self._get_jump_transition_prob(prev_m_state, next_m_state)
                p_trans_l = self._get_leverage_transition_prob(prev_l_state, next_l_state, current_trend_for_leverage)
                A_t[i_idx, j_idx] = p_trans_c * p_trans_m * p_trans_l
        row_sums = A_t.sum(axis=1, keepdims=True)
        A_t = np.divide(A_t, row_sums, out=np.zeros_like(A_t), where=row_sums > 1e-15) # Increased tolerance
        if np.any(row_sums <= 1e-15):
            for i in range(self.num_combined_fhmv_states):
                if row_sums[i] <= 1e-15: A_t[i, :] = 1.0 / self.num_combined_fhmv_states
        return A_t

    # --- E-Step and M-Step ---
    # (_e_step, _update_dof_component_jk, _update_fhmv_transition_parameters,
    #  _q_emissions_objective, _update_fhmv_magnitude_parameters, _m_step, train
    #  are now assumed to be present and use the ECM logic correctly with Sigma_BASE.
    #  The code for these was provided in the previous approved response.
    #  I will include the refined _m_step and its helpers here for completeness of this file.)

    def _e_step(self, features_df: pd.DataFrame):
        # (Code from previous approved response, with corrected loop vars and using current_fhmv_mag_params in _get_emission_log_probabilities)
        features_np = features_df.values 
        T = features_np.shape[0]; N_states = self.num_combined_fhmv_states; K_mix = self.num_stmm_mix_components
        
        # Use FHMV magnitude params from self.fhmv_params_ (i.e. from previous M-step or init)
        log_b_jt = self._get_emission_log_probabilities(features_np, self.fhmv_params_) 
        
        log_alpha_t = np.full((T, N_states), -np.inf)
        log_initial_state_probs = np.log(self.initial_state_probs_ + 1e-30) # Ensure initial_state_probs_ is not log
        log_alpha_t[0, :] = log_initial_state_probs + log_b_jt[0, :]
        
        for t_loop_alpha in range(1, T): # Changed loop variable
            # A(t-1) = P(S_t | S_{t-1}), use Trend at t-1. features_df is 0-indexed.
            A_t_minus_1 = self._get_combined_transition_matrix_A_t(t_loop_alpha - 1, features_df)
            log_A_t_minus_1 = np.log(A_t_minus_1 + 1e-30)
            for j_idx in range(N_states): 
                prev_alpha_plus_log_A_ij = log_alpha_t[t_loop_alpha-1, :] + log_A_t_minus_1[:, j_idx] 
                max_val = np.max(prev_alpha_plus_log_A_ij)
                if np.isinf(max_val): log_sum_exp_val = -np.inf
                else: log_sum_exp_val = max_val + np.log(np.sum(np.exp(prev_alpha_plus_log_A_ij - max_val)))
                log_alpha_t[t_loop_alpha, j_idx] = log_sum_exp_val + log_b_jt[t_loop_alpha, j_idx]
        
        current_log_likelihood = self._log_likelihood(log_alpha_t)

        log_beta_t = np.full((T, N_states), -np.inf); log_beta_t[T-1, :] = 0.0 
        for t_loop_beta in range(T-2, -1, -1): # Changed loop variable
            # A(t) = P(S_{t+1} | S_t), use Trend at t.
            A_t = self._get_combined_transition_matrix_A_t(t_loop_beta, features_df)
            log_A_t = np.log(A_t + 1e-30)
            for i_idx in range(N_states): 
                terms_for_sum_j = log_A_t[i_idx, :] + log_b_jt[t_loop_beta+1, :] + log_beta_t[t_loop_beta+1, :]
                max_val = np.max(terms_for_sum_j)
                if np.isinf(max_val): log_beta_t[t_loop_beta, i_idx] = -np.inf
                else: log_beta_t[t_loop_beta, i_idx] = max_val + np.log(np.sum(np.exp(terms_for_sum_j - max_val)))
        
        log_gamma_t = log_alpha_t + log_beta_t - current_log_likelihood
        gamma_t = np.exp(log_gamma_t)
        gamma_t = gamma_t / (np.sum(gamma_t, axis=1, keepdims=True) + 1e-30)

        log_xi_t = np.full((T-1, N_states, N_states), -np.inf)
        for t_idx_xi in range(T-1): # Changed loop variable 't_idx' to 't_idx_xi'
            A_t_for_xi = self._get_combined_transition_matrix_A_t(t_idx_xi, features_df)
            log_A_t_for_xi = np.log(A_t_for_xi + 1e-30)
            for i_state in range(N_states):
                for j_state in range(N_states):
                    val = log_alpha_t[t_idx_xi, i_state] + log_A_t_for_xi[i_state, j_state] + \
                          log_b_jt[t_idx_xi+1, j_state] + log_beta_t[t_idx_xi+1, j_state] - \
                          current_log_likelihood
                    if np.isfinite(val): log_xi_t[t_idx_xi, i_state, j_state] = val
        xi_t = np.exp(log_xi_t)
        for t_idx_xi_norm in range(T-1): # Changed loop variable
             xi_t[t_idx_xi_norm, :, :] = xi_t[t_idx_xi_norm, :, :] / (np.sum(xi_t[t_idx_xi_norm, :, :]) + 1e-30)
        
        tau_t_jk = np.zeros((T, N_states, K_mix)); E_w_t_jk = np.zeros((T, N_states, K_mix)); E_log_w_t_jk = np.zeros((T, N_states, K_mix))
        for t_loop_tau in range(T): # Changed loop variable
            O_t = features_np[t_loop_tau,:]
            for j_state_idx_tau in range(N_states): # Changed loop variable
                if gamma_t[t_loop_tau,j_state_idx_tau] < 1e-20: continue
                stmm_params_state_j = self.stmm_emission_params_[j_state_idx_tau]; component_pdfs_state_j = np.zeros(K_mix)
                
                # Use current FHMV magnitude parameters for calculating P(Z_k | S_j, O_t) part of tau_t_jk
                # and for E_w, E_log_w which depend on Sigma_eff_old and nu_old
                fhmv_mag_params_current_iter = self.fhmv_params_ 

                for k_idx in range(K_mix):
                    comp_k_params = stmm_params_state_j[k_idx]
                    sigma_effective_jk = self._get_effective_covariance(j_state_idx_tau, k_idx, fhmv_mag_params_current_iter)
                    component_pdfs_state_j[k_idx] = self._multivariate_student_t_pdf(O_t, comp_k_params['mean'], sigma_effective_jk, comp_k_params['dof'], log_pdf=False)
                
                weighted_pdfs = np.array([stmm_params_state_j[k_idx]['weight'] * component_pdfs_state_j[k_idx] for k_idx in range(K_mix)])
                sum_weighted_pdfs = np.sum(weighted_pdfs)

                if sum_weighted_pdfs > 1e-30: 
                    prob_Z_is_k_given_S_is_j_Ot = weighted_pdfs / sum_weighted_pdfs
                    tau_t_jk[t_loop_tau,j_state_idx_tau,:] = gamma_t[t_loop_tau,j_state_idx_tau] * prob_Z_is_k_given_S_is_j_Ot
                else: 
                    tau_t_jk[t_loop_tau,j_state_idx_tau,:] = (gamma_t[t_loop_tau,j_state_idx_tau] / K_mix) if K_mix > 0 else 0.0
                
                for k_idx in range(K_mix):
                    comp_k_params = stmm_params_state_j[k_idx]; mu_jk = comp_k_params['mean']
                    sigma_effective_jk_old = self._get_effective_covariance(j_state_idx_tau, k_idx, fhmv_mag_params_current_iter)
                    try: Sigma_jk_eff_inv = inv(sigma_effective_jk_old)
                    except np.linalg.LinAlgError: Sigma_jk_eff_inv = pinv(sigma_effective_jk_old)
                    nu_jk_old = comp_k_params['dof']; D_feat = self.num_features
                    delta_sq_t_jk = (O_t - mu_jk).T @ Sigma_jk_eff_inv @ (O_t - mu_jk)
                    if delta_sq_t_jk < 0: delta_sq_t_jk = 0 # Defensive
                    
                    current_E_w = (nu_jk_old + D_feat) / (nu_jk_old + delta_sq_t_jk + 1e-20)
                    E_w_t_jk[t_loop_tau,j_state_idx_tau,k_idx] = current_E_w
                    E_log_w_t_jk[t_loop_tau,j_state_idx_tau,k_idx] = digamma((nu_jk_old + D_feat) / 2.0) - \
                                                                  np.log((nu_jk_old + D_feat) / 2.0 + 1e-30) + \
                                                                  np.log(current_E_w + 1e-30) 
        return {'log_alpha_t': log_alpha_t, 'log_beta_t': log_beta_t, 'gamma_t': gamma_t, 'xi_t': xi_t, 
                'tau_t_jk': tau_t_jk, 'E_w_t_jk': E_w_t_jk, 'E_log_w_t_jk': E_log_w_t_jk, 
                'current_log_likelihood': current_log_likelihood}

    def _update_dof_component_jk(self, nu_jk_old: float, N_jk_eff: float, sum_tau_E_ln_w_jk: float, sum_tau_E_w_jk:float):
        """
        Enhanced DoF parameter estimation with improved numerical stability and convergence.
        """
        # Enhanced threshold checking
        if N_jk_eff < 1e-6 or np.isnan(N_jk_eff) or np.isinf(N_jk_eff):
            return nu_jk_old
            
        # Robust calculation of C_jk_term with outlier detection
        try:
            C_jk_term = (sum_tau_E_ln_w_jk - sum_tau_E_w_jk) / N_jk_eff
            
            # Check for extreme values that might indicate numerical issues
            if np.isnan(C_jk_term) or np.isinf(C_jk_term) or abs(C_jk_term) > 100:
                return nu_jk_old
                
        except (ZeroDivisionError, OverflowError):
            return nu_jk_old
            
        D_FEATURES = self.num_features
        
        def objective_func_for_nu(nu_opt):
            """Enhanced objective function with better numerical handling."""
            if nu_opt <= 1e-3: 
                return 1e12
                
            try:
                # Use more robust numerical calculations
                term1 = -digamma(nu_opt / 2.0) + np.log(nu_opt / 2.0 + 1e-30) + 1.0
                term2 = digamma((nu_opt + D_FEATURES) / 2.0) - np.log((nu_opt + D_FEATURES) / 2.0 + 1e-30)
                result = term1 - term2 + C_jk_term
                
                # Clamp result to prevent extreme values
                return np.clip(result, -1e10, 1e10)
                
            except (OverflowError, ValueError):
                return 1e12 if nu_opt < nu_jk_old else -1e12
        
        try:
            # Enhanced bounds checking with adaptive adjustment
            dof_min_adaptive = max(self.dof_min, 2.01)
            dof_max_adaptive = min(self.dof_max, 100.0)  # Cap at reasonable value
            
            f_min = objective_func_for_nu(dof_min_adaptive)
            f_max = objective_func_for_nu(dof_max_adaptive)
            
            # More robust NaN/Inf checking
            if not all(np.isfinite([f_min, f_max])):
                return nu_jk_old
                
            # Check if root exists in interval
            if np.sign(f_min) == np.sign(f_max):
                # Return the bound with smaller absolute objective value
                return dof_min_adaptive if abs(f_min) < abs(f_max) else dof_max_adaptive
            
            # Enhanced root finding with multiple fallbacks
            try:
                # Primary: Brent's method
                new_nu = brentq(objective_func_for_nu, dof_min_adaptive, dof_max_adaptive, 
                               disp=False, xtol=1e-6, rtol=1e-6, maxiter=50)
                               
                # Validate result
                if dof_min_adaptive <= new_nu <= dof_max_adaptive and np.isfinite(new_nu):
                    return new_nu
                else:
                    return nu_jk_old
                    
            except (ValueError, RuntimeError):
                # Fallback: Bisection method
                try:
                    a, b = dof_min_adaptive, dof_max_adaptive
                    for _ in range(20):  # Limited iterations
                        c = (a + b) / 2
                        fc = objective_func_for_nu(c)
                        if abs(fc) < 1e-6:
                            return c
                        if np.sign(fc) == np.sign(f_min):
                            a = c
                        else:
                            b = c
                    return (a + b) / 2
                except:
                    return nu_jk_old
                    
        except Exception:
            return nu_jk_old

    def _update_fhmv_transition_parameters(self, T: int, xi_t: np.ndarray, gamma_t: np.ndarray):
        # (Code from previous approved response)
        new_p_persistence = np.zeros(self.num_persistence_chains)
        for n in range(self.num_persistence_chains):
            expected_stays_chain_n_s0 = 0.0; expected_stays_chain_n_s1 = 0.0
            expected_in_chain_n_s0_at_t = 0.0; expected_in_chain_n_s1_at_t = 0.0
            for t_loop in range(T - 1): # Corrected loop variable
                for i_idx in range(self.num_combined_fhmv_states):
                    prev_comp_states = self._idx_to_comp_states[i_idx]
                    prev_p_chain_n_state = prev_comp_states[0][n]
                    if prev_p_chain_n_state == 0: expected_in_chain_n_s0_at_t += gamma_t[t_loop, i_idx]
                    else: expected_in_chain_n_s1_at_t += gamma_t[t_loop, i_idx]
                    for j_idx in range(self.num_combined_fhmv_states):
                        next_comp_states = self._idx_to_comp_states[j_idx]
                        next_p_chain_n_state = next_comp_states[0][n]
                        if prev_p_chain_n_state == 0 and next_p_chain_n_state == 0: expected_stays_chain_n_s0 += xi_t[t_loop, i_idx, j_idx]
                        elif prev_p_chain_n_state == 1 and next_p_chain_n_state == 1: expected_stays_chain_n_s1 += xi_t[t_loop, i_idx, j_idx]
            total_expected_stays = expected_stays_chain_n_s0 + expected_stays_chain_n_s1
            total_expected_time_at_t = expected_in_chain_n_s0_at_t + expected_in_chain_n_s1_at_t
            if total_expected_time_at_t > 1e-20:
                new_p_persistence[n] = np.clip(total_expected_stays / total_expected_time_at_t, self.min_prob_transition_param, self.max_prob_transition_param)
            else: new_p_persistence[n] = self.fhmv_params_['p_persistence'][n]
        self.fhmv_params_['p_persistence'] = new_p_persistence
        if self.num_jump_states == 2:
            expected_m0_to_m1 = 0.0; expected_m1_to_m0 = 0.0
            expected_in_m0_at_t = 0.0; expected_in_m1_at_t = 0.0
            for t_loop in range(T - 1): # Corrected loop variable
                for i_idx in range(self.num_combined_fhmv_states):
                    prev_comp_states = self._idx_to_comp_states[i_idx]; prev_m_state = prev_comp_states[1]
                    if prev_m_state == 0: expected_in_m0_at_t += gamma_t[t_loop, i_idx]
                    else: expected_in_m1_at_t += gamma_t[t_loop, i_idx]
                    for j_idx in range(self.num_combined_fhmv_states):
                        next_comp_states = self._idx_to_comp_states[j_idx]; next_m_state = next_comp_states[1]
                        if prev_m_state == 0 and next_m_state == 1: expected_m0_to_m1 += xi_t[t_loop, i_idx, j_idx]
                        elif prev_m_state == 1 and next_m_state == 0: expected_m1_to_m0 += xi_t[t_loop, i_idx, j_idx]
            if expected_in_m0_at_t > 1e-20: self.fhmv_params_['q_jump'] = np.clip(expected_m0_to_m1 / expected_in_m0_at_t, self.min_prob_transition_param, self.max_prob_transition_param)
            if expected_in_m1_at_t > 1e-20: self.fhmv_params_['q_jump_revert'] = np.clip(expected_m1_to_m0 / expected_in_m1_at_t, self.min_prob_transition_param, self.max_prob_transition_param)


    def _q_emissions_objective(self, fhmv_mag_params_array: np.ndarray,
                               features_np: np.ndarray, 
                               gamma_t: np.ndarray, # Kept for interface consistency but not used in current implementation
                               tau_t_jk_from_E_step: np.ndarray, # P(S_t=j, Z_tj=k | O)
                               base_stmm_params_for_opt: list # Contains mu, Sigma_jk_BASE, nu, weight
                               ) -> float:
        # (Code from previous approved response - uses tau_t_jk_from_E_step for Q-func)
        current_fhmv_mag_params_dict = {
            'c1_persistence': fhmv_mag_params_array[0], 'theta_c_persistence': fhmv_mag_params_array[1],
            'm1_jump': fhmv_mag_params_array[2], 'theta_m_jump': fhmv_mag_params_array[3],
            'l1_leverage': fhmv_mag_params_array[4], 'theta_l_leverage': fhmv_mag_params_array[5],
            'sigma_sq_overall': fhmv_mag_params_array[6]
        }
        T = features_np.shape[0]; N_states = self.num_combined_fhmv_states; K_mix = self.num_stmm_mix_components
        total_expected_log_likelihood_term = 0.0
        for t_loop in range(T): # Changed loop variable
            O_t = features_np[t_loop, :]
            for j_idx in range(N_states):
                # The factor C_j, M_j, L_j depends only on the combined state j_idx and mag_params
                C_j, M_j, L_j = self._calculate_fhmv_factors_for_state(j_idx, current_fhmv_mag_params_dict)
                sigma_sq = current_fhmv_mag_params_dict['sigma_sq_overall']
                effective_variance_scale = np.abs(sigma_sq * C_j * M_j * L_j) + 1e-12 # Epsilon for stability
                if effective_variance_scale < 1e-12: effective_variance_scale = 1e-12

                state_j_base_stmm_params = base_stmm_params_for_opt[j_idx]
                for k_idx in range(K_mix):
                    if tau_t_jk_from_E_step[t_loop, j_idx, k_idx] > 1e-20: # If this component has responsibility
                        comp_k_base_params = state_j_base_stmm_params[k_idx]
                        mu_jk = comp_k_base_params['mean']
                        Sigma_jk_base = comp_k_base_params['covariance'] # This is Sigma_BASE
                        nu_jk = comp_k_base_params['dof']
                        
                        Sigma_jk_effective = effective_variance_scale * Sigma_jk_base
                        
                        log_pdf_val = self._multivariate_student_t_pdf(
                            O_t, mu_jk, Sigma_jk_effective, nu_jk, log_pdf=True
                        )
                        if not np.isinf(log_pdf_val) and not np.isnan(log_pdf_val):
                            total_expected_log_likelihood_term += tau_t_jk_from_E_step[t_loop, j_idx, k_idx] * log_pdf_val
        # print(f"    Q_emissions for params {fhmv_mag_params_array}: {-total_expected_log_likelihood_term:.4f}")
        return -total_expected_log_likelihood_term 

    def _update_fhmv_magnitude_parameters(self, features_np: np.ndarray, e_step_outputs: dict):
        # (Code from previous approved response - now makes the call to minimize)
        print("M-Step: Updating FHMV Magnitude Parameters (Numerical Optimization)...")
        initial_guess = np.array([
            self.fhmv_params_['c1_persistence'], self.fhmv_params_['theta_c_persistence'],
            self.fhmv_params_['m1_jump'], self.fhmv_params_['theta_m_jump'],
            self.fhmv_params_['l1_leverage'], self.fhmv_params_['theta_l_leverage'],
            self.fhmv_params_['sigma_sq_overall']
        ])
        bounds_list = [
            self.fhmv_magnitude_param_bounds['c1_persistence'], self.fhmv_magnitude_param_bounds['theta_c_persistence'],
            self.fhmv_magnitude_param_bounds['m1_jump'], self.fhmv_magnitude_param_bounds['theta_m_jump'],
            self.fhmv_magnitude_param_bounds['l1_leverage'], self.fhmv_magnitude_param_bounds['theta_l_leverage'],
            self.fhmv_magnitude_param_bounds['sigma_sq_overall']
        ]
        # Ensure initial guess for sigma_sq is positive and within bounds, as it's critical
        initial_guess[6] = np.clip(initial_guess[6], bounds_list[6][0] + 1e-9, bounds_list[6][1] - 1e-9)
        if initial_guess[6] <= 0 : initial_guess[6] = bounds_list[6][0] + 1e-9

        args_for_obj = (features_np, 
                        e_step_outputs['gamma_t'], # gamma_t might not be strictly needed if using tau_t_jk correctly
                        e_step_outputs['tau_t_jk'],# P(S_t=j, Z_tj=k | O)
                        self.stmm_emission_params_ # Contains mu, Sigma_BASE, nu, weight from StMM M-step part 1
                       )
        result = minimize(self._q_emissions_objective, 
                          initial_guess, 
                          args=args_for_obj, 
                          method='L-BFGS-B', 
                          bounds=bounds_list,
                          options={'maxiter': 25, 'ftol': 1e-7, 'eps':1e-8}) # Optimizer settings

        if result.success or (result.status in [2] and result.fun < self._q_emissions_objective(initial_guess, *args_for_obj) * 0.999 ): # Status 2 for precision loss
            optimized_params = result.x
            self.fhmv_params_['c1_persistence'] = optimized_params[0]
            self.fhmv_params_['theta_c_persistence'] = optimized_params[1]
            self.fhmv_params_['m1_jump'] = optimized_params[2]
            self.fhmv_params_['theta_m_jump'] = optimized_params[3]
            self.fhmv_params_['l1_leverage'] = optimized_params[4]
            self.fhmv_params_['theta_l_leverage'] = optimized_params[5]
            self.fhmv_params_['sigma_sq_overall'] = optimized_params[6]
            print(f"  Magnitude parameters updated. Objective: {result.fun:.4f}")
        else:
            print(f"  Magnitude parameter optimization FAILED or did not improve significantly. Status: {result.status}, Msg: {result.message}. Keeping old values.")
            # print(f"    Initial Objective: {self._q_emissions_objective(initial_guess, *args_for_obj):.4f}, Final Objective: {result.fun:.4f}")


    def _m_step(self, features_df: pd.DataFrame, e_step_outputs: dict):
        # (Code from previous approved response, incorporating ECM logic for Sigma_BASE)
        features_np = features_df.values; T = features_np.shape[0]
        gamma_t = e_step_outputs['gamma_t']; xi_t = e_step_outputs['xi_t']
        tau_t_jk = e_step_outputs['tau_t_jk'] 
        E_w_t_jk = e_step_outputs['E_w_t_jk']; E_log_w_t_jk = e_step_outputs['E_log_w_t_jk']
        
        # --- ECM Step 1: Update StMM base parameters & FHMV transition params ---
        self.initial_state_probs_ = gamma_t[0, :] / (np.sum(gamma_t[0, :]) + 1e-30)
        self._update_fhmv_transition_parameters(T, xi_t, gamma_t)

        new_stmm_emission_params = [] 
        N_states = self.num_combined_fhmv_states; K_mix = self.num_stmm_mix_components
        fhmv_mag_params_at_m_step_start = self.fhmv_params_.copy() # Params from prev EM iter / start of current M

        for j in range(N_states):
            state_j_gamma_sum = np.sum(gamma_t[:, j]) + 1e-30; new_state_j_components = []
            C_j_old, M_j_old, L_j_old = self._calculate_fhmv_factors_for_state(j, fhmv_mag_params_at_m_step_start)
            sigma_sq_old = fhmv_mag_params_at_m_step_start['sigma_sq_overall']
            current_total_variance_scale_factor_j = np.abs(sigma_sq_old * C_j_old * M_j_old * L_j_old) + 1e-12
            if current_total_variance_scale_factor_j < 1e-12: current_total_variance_scale_factor_j = 1e-12

            for k in range(K_mix):
                tau_jk_sum = np.sum(tau_t_jk[:, j, k]) + 1e-30
                new_w_jk = tau_jk_sum / state_j_gamma_sum
                numerator_mu = np.zeros(self.num_features); denominator_mu = 1e-30 
                for t_loop in range(T):
                    if tau_t_jk[t_loop,j,k] > 1e-20 : 
                        weighted_obs = E_w_t_jk[t_loop,j,k] * features_np[t_loop,:]
                        numerator_mu += tau_t_jk[t_loop,j,k] * weighted_obs
                        denominator_mu += tau_t_jk[t_loop,j,k] * E_w_t_jk[t_loop,j,k]
                new_mu_jk = numerator_mu / denominator_mu
                if np.isnan(new_mu_jk).any(): new_mu_jk = self.stmm_emission_params_[j][k]['mean']
                
                numerator_Sigma_base = np.zeros((self.num_features, self.num_features))
                for t_loop in range(T):
                     if tau_t_jk[t_loop,j,k] > 1e-20:
                        diff = features_np[t_loop,:] - new_mu_jk; outer_prod = np.outer(diff, diff)
                        numerator_Sigma_base += tau_t_jk[t_loop,j,k] * E_w_t_jk[t_loop,j,k] * outer_prod / current_total_variance_scale_factor_j
                new_Sigma_jk_base = numerator_Sigma_base / (tau_jk_sum + 1e-30) 
                new_Sigma_jk_base = new_Sigma_jk_base + np.eye(self.num_features) * 1e-9 # Regularize Sigma_BASE
                if np.isnan(new_Sigma_jk_base).any().any(): new_Sigma_jk_base = self.stmm_emission_params_[j][k]['covariance'] 

                nu_jk_old = self.stmm_emission_params_[j][k]['dof']
                sum_tau_E_ln_w_jk_val = np.sum(tau_t_jk[:,j,k] * E_log_w_t_jk[:,j,k])
                sum_tau_E_w_jk_val = np.sum(tau_t_jk[:,j,k] * E_w_t_jk[:,j,k])
                N_jk_eff_val = tau_jk_sum
                new_nu_jk = self._update_dof_component_jk(nu_jk_old, N_jk_eff_val, sum_tau_E_ln_w_jk_val, sum_tau_E_w_jk_val)
                
                new_state_j_components.append({'weight': new_w_jk, 'mean': new_mu_jk, 
                                               'covariance': new_Sigma_jk_base, # Store updated SIGMA_BASE
                                               'dof': new_nu_jk})
            # ... (Normalize weights) ...
            current_sum_weights = sum(c['weight'] for c in new_state_j_components); K_mix_eff = K_mix if K_mix > 0 else 1.0
            if current_sum_weights > 1e-20: [comp.update(weight = comp['weight']/current_sum_weights) for comp in new_state_j_components]
            else: [comp.update(weight = 1.0/K_mix_eff) for comp in new_state_j_components]
            new_stmm_emission_params.append(new_state_j_components)
        self.stmm_emission_params_ = new_stmm_emission_params 

        # --- ECM Step 2: Update FHMV Magnitude Parameters ---
        self._update_fhmv_magnitude_parameters(features_np, e_step_outputs) 
        
        # Nudge DoF off boundaries
        for j in range(N_states): 
            for k in range(K_mix):
                dof_val = self.stmm_emission_params_[j][k]['dof']
                if dof_val <= self.dof_min : self.stmm_emission_params_[j][k]['dof'] = self.dof_min + 1e-3
                if dof_val >= self.dof_max : self.stmm_emission_params_[j][k]['dof'] = self.dof_max - 1e-3

    # --- Train and Inference Methods ---
    def train(self, training_features: pd.DataFrame):
        # (Code from previous approved response)
        if not isinstance(training_features, pd.DataFrame) or training_features.empty: raise ValueError("training_features must be non-empty DF.")
        if training_features.shape[1] != self.num_features: raise ValueError(f"training_features must have {self.num_features} columns.")
        
        print("Starting FHMV Training Sub-Module...");
        self._initialize_all_parameters(training_features)

        previous_log_likelihood = -np.inf 
        print(f"Beginning EM algorithm for up to {self.em_max_iterations} iterations...")
        
        converged = False
        for iteration in range(self.em_max_iterations):
            print(f"\n--- EM Iteration: {iteration + 1}/{self.em_max_iterations} ---")
            print("Performing E-Step...");
            try: 
                e_step_outputs = self._e_step(training_features)
            except np.linalg.LinAlgError as e: print(f"LinAlgError in E-step: {e}. Stopping."); break
            except Exception as e: print(f"Error in E-step: {e}. Stopping."); traceback.print_exc(); break
            
            current_log_likelihood = e_step_outputs['current_log_likelihood']
            print(f"Log-Likelihood: {current_log_likelihood:.6f}")
            if np.isnan(current_log_likelihood) or np.isinf(current_log_likelihood): print("LL is NaN/Inf. Stopping."); break
            
            if iteration > 0 and abs(current_log_likelihood - previous_log_likelihood) < self.em_tolerance : 
                print(f"EM Algorithm Converged after {iteration + 1} iterations."); converged = True; break
            if iteration > 0 and current_log_likelihood < previous_log_likelihood - abs(previous_log_likelihood * 1e-5) : 
                 print(f"Warning: LL decreased from {previous_log_likelihood:.4f} to {current_log_likelihood:.4f}.") # Could add more logic here
            previous_log_likelihood = current_log_likelihood
            
            print("Performing M-Step...");
            try: 
                self._m_step(training_features, e_step_outputs)
            except np.linalg.LinAlgError as e: print(f"LinAlgError in M-step: {e}. Stopping."); break
            except Exception as e: print(f"Error in M-step: {e}. Stopping."); traceback.print_exc(); break
        
        if not converged and iteration == self.em_max_iterations - 1 : print(f"EM reached max iterations ({self.em_max_iterations}) without specified convergence.")
        print("\nFHMV Training Finished.")

    def get_most_likely_state_sequence(self, features_df: pd.DataFrame) -> np.ndarray:
        # (Code from previous approved response, ensure A_t uses features_df for Trend)
        if self.stmm_emission_params_ is None or self.initial_state_probs_ is None: raise RuntimeError("Model not trained.")
        if not isinstance(features_df, pd.DataFrame): raise TypeError("features_df must be a Pandas DataFrame.")
        features_np = features_df.values; T = features_np.shape[0]; N_states = self.num_combined_fhmv_states
        log_b_jt = self._get_emission_log_probabilities(features_np, self.fhmv_params_) # Pass current FHMV mag params
        log_delta_t = np.full((T, N_states), -np.inf); psi_t = np.zeros((T, N_states), dtype=int)
        log_initial_state_probs = np.log(self.initial_state_probs_ + 1e-30)
        log_delta_t[0, :] = log_initial_state_probs + log_b_jt[0, :]
        for t in range(1, T):
            A_t_minus_1 = self._get_combined_transition_matrix_A_t(t - 1, features_df) 
            log_A_t_minus_1 = np.log(A_t_minus_1 + 1e-30)
            for j_idx in range(N_states): # Corrected loop variable
                log_probs_from_prev_state = log_delta_t[t-1, :] + log_A_t_minus_1[:, j_idx]
                if len(log_probs_from_prev_state) == 0: log_delta_t[t,j_idx] = -np.inf; psi_t[t,j_idx] = 0
                else:
                    max_log_prob = np.max(log_probs_from_prev_state)
                    psi_t[t,j_idx] = np.argmax(log_probs_from_prev_state) if len(log_probs_from_prev_state)>0 else 0
                    if np.isinf(max_log_prob): log_delta_t[t,j_idx] = -np.inf; 
                    else: log_delta_t[t,j_idx] = max_log_prob + log_b_jt[t,j_idx]; 
        most_likely_path = np.zeros(T, dtype=int)
        if T > 0:
            most_likely_path[T-1] = np.argmax(log_delta_t[T-1, :])
            for t_path in range(T-2, -1, -1): most_likely_path[t_path] = psi_t[t_path+1, most_likely_path[t_path+1]]
        if self.combined_to_final_regime_mapping:
            mapped_path = np.array([self.combined_to_final_regime_mapping.get(state_idx, -1) for state_idx in most_likely_path])
            if np.any(mapped_path == -1): print("Warning: Viterbi path unmapped states.")
            return mapped_path
        else: return most_likely_path

    def get_state_probabilities(self, features_df: pd.DataFrame, smoothed: bool = True) -> np.ndarray:
        # (Code from previous approved response, ensure A_t uses features_df for Trend)
        if self.stmm_emission_params_ is None or self.initial_state_probs_ is None: raise RuntimeError("Model not trained.")
        if smoothed:
            e_step_outputs = self._e_step(features_df); combined_state_probs = e_step_outputs['gamma_t']
        else:
            features_np = features_df.values; T = features_np.shape[0]; N_states = self.num_combined_fhmv_states
            log_b_jt = self._get_emission_log_probabilities(features_np, self.fhmv_params_) # Pass current FHMV mag params
            log_alpha_t = np.full((T, N_states), -np.inf)
            log_initial_state_probs = np.log(self.initial_state_probs_ + 1e-30)
            log_alpha_t[0, :] = log_initial_state_probs + log_b_jt[0, :]
            for t in range(1, T):
                A_t_minus_1 = self._get_combined_transition_matrix_A_t(t - 1, features_df) 
                log_A_t_minus_1 = np.log(A_t_minus_1 + 1e-30)
                for j_idx in range(N_states): # Corrected loop variable
                    prev_alpha_plus_log_A_ij = log_alpha_t[t-1, :] + log_A_t_minus_1[:, j_idx]
                    max_val = np.max(prev_alpha_plus_log_A_ij)
                    if np.isinf(max_val): log_sum_exp_val = -np.inf
                    else: log_sum_exp_val = max_val + np.log(np.sum(np.exp(prev_alpha_plus_log_A_ij - max_val)))
                    log_alpha_t[t, j_idx] = log_sum_exp_val + log_b_jt[t, j_idx]
            log_sum_alpha_t_per_step = np.zeros(T)
            for t_loop in range(T): # Corrected loop variable
                max_log = np.max(log_alpha_t[t_loop, :])
                if np.isinf(max_log): log_sum_alpha_t_per_step[t_loop] = -np.inf
                else: log_sum_alpha_t_per_step[t_loop] = max_log + np.log(np.sum(np.exp(log_alpha_t[t_loop, :] - max_log)))
            log_filtered_probs = log_alpha_t - log_sum_alpha_t_per_step[:, np.newaxis]
            combined_state_probs = np.exp(log_filtered_probs)
            combined_state_probs = combined_state_probs / (np.sum(combined_state_probs, axis=1, keepdims=True) + 1e-30)
        if self.combined_to_final_regime_mapping:
            final_regime_probs = np.zeros((features_df.shape[0], self.num_fhmv_regimes))
            for combined_idx, final_regime_idx in self.combined_to_final_regime_mapping.items():
                if 0 <= combined_idx < combined_state_probs.shape[1]:
                    final_regime_probs[:, final_regime_idx] += combined_state_probs[:, combined_idx]
            final_regime_probs = final_regime_probs / (np.sum(final_regime_probs, axis=1, keepdims=True) + 1e-30)
            return final_regime_probs
        else: return combined_state_probs

    def _get_emission_log_probabilities(self, features_np: np.ndarray, fhmv_mag_params: dict) -> np.ndarray:
        """
        Calculates emission log-probabilities for all time steps and states.
        
        Args:
            features_np: T x D array of feature observations
            fhmv_mag_params: Dictionary containing FHMV magnitude parameters
            
        Returns:
            log_b_jt: T x N_states array of log emission probabilities
        """
        T, _ = features_np.shape  # D not used in this method
        N_states = self.num_combined_fhmv_states
        K_mix = self.num_stmm_mix_components
        
        log_b_jt = np.full((T, N_states), -np.inf)
        
        for t in range(T):
            O_t = features_np[t, :]
            for j in range(N_states):
                # Calculate mixture probabilities for state j
                log_mixture_probs = np.full(K_mix, -np.inf)
                
                for k in range(K_mix):
                    comp_params = self.stmm_emission_params_[j][k]
                    weight_jk = comp_params['weight']
                    mu_jk = comp_params['mean']
                    dof_jk = comp_params['dof']
                    
                    # Get effective covariance using FHMV factors
                    sigma_eff_jk = self._get_effective_covariance(j, k, fhmv_mag_params)
                    
                    # Calculate log PDF
                    log_pdf = self._multivariate_student_t_pdf(
                        O_t, mu_jk, sigma_eff_jk, dof_jk, log_pdf=True
                    )
                    
                    # Add log weight
                    if weight_jk > 1e-30:
                        log_mixture_probs[k] = np.log(weight_jk) + log_pdf
                
                # Log-sum-exp for numerical stability
                max_log_prob = np.max(log_mixture_probs)
                if not np.isinf(max_log_prob):
                    log_b_jt[t, j] = max_log_prob + np.log(
                        np.sum(np.exp(log_mixture_probs - max_log_prob))
                    )
                
        return log_b_jt
    
    def _log_likelihood(self, log_alpha_t: np.ndarray) -> float:
        """
        Calculates the log-likelihood of the observation sequence.
        
        Args:
            log_alpha_t: T x N_states array of forward probabilities in log space
            
        Returns:
            Log-likelihood of the observation sequence
        """
        T = log_alpha_t.shape[0]
        if T == 0:
            return -np.inf
            
        # Log-sum-exp over final time step
        log_alpha_final = log_alpha_t[T-1, :]
        max_log_alpha = np.max(log_alpha_final)
        
        if np.isinf(max_log_alpha):
            return -np.inf
            
        log_likelihood = max_log_alpha + np.log(
            np.sum(np.exp(log_alpha_final - max_log_alpha))
        )
        
        return log_likelihood