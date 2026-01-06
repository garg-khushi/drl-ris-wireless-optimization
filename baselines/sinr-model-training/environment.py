import numpy as np
#from impulsive_noise import impulsive_noise_generation
import numpy as np

def impulsive_noise_generation_np(N, lambda_, sigma0, sigma1):
    """
    NumPy version of impulsive noise generator
    """
    # Background Gaussian noise (complex)
    w_k = (sigma0 / np.sqrt(2)) * (
        np.random.randn(N) + 1j * np.random.randn(N)
    )
    
    # Bernoulli sequence
    b_k = (np.random.rand(N) < lambda_).astype(float)
    
    # Impulsive Gaussian noise (complex)
    g_k = (sigma1 / np.sqrt(2)) * (
        np.random.randn(N) + 1j * np.random.randn(N)
    )
    
    return w_k + b_k * g_k
class RIS_MISO(object):
    def __init__(self,
                 num_antennas,
                 num_RIS_elements,
                 num_users,
                 channel_est_error=True,
                 AWGN_var=1e-2,
                 channel_noise_var=1e-2,
                 impulsive_lambda=0.01,
                 impulsive_sigma0=0.001,
                 impulsive_sigma1=0.01):

        self.M = num_antennas
        self.L = num_RIS_elements
        self.K = num_users

        self.channel_est_error = channel_est_error
        # New impulsive noise parameters
        self.impulsive_lambda = impulsive_lambda
        self.impulsive_sigma0 = impulsive_sigma0
        self.impulsive_sigma1 = impulsive_sigma1
        assert self.M == self.K

        self.awgn_var = AWGN_var
        self.channel_noise_var = channel_noise_var

        power_size = 2 * self.K

        channel_size = 2 * (self.L * self.M + self.L * self.K)

        self.action_dim = 2 * self.M * self.K + 2 * self.L
        self.state_dim = power_size + channel_size + self.action_dim

        self.H_1 = None
        self.H_2 = None
        self.G = np.eye(self.M, dtype=complex)
        self.Phi = np.eye(self.L, dtype=complex)

        self.state = None
        self.done = None

        self.episode_t = None
    
    def _compute_H_2_tilde(self):
        return self.H_2.T @ self.Phi @ self.H_1 @ self.G

    def reset(self):
        self.episode_t = 0

        self.H_1 = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) + 1j * np.random.normal(0, np.sqrt(0.5),
                                                                                               (self.L, self.M))
        self.H_2 = np.random.normal(0, np.sqrt(0.5), (self.L, self.K)) + 1j * np.random.normal(0, np.sqrt(0.5),
                                                                                        (self.L, self.K))
        self.H_1_true = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.M))
        self.H_2_true = np.random.normal(0, np.sqrt(0.5), (self.L, self.K)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.L, self.K))
        self.H_1 = self.H_1_true.copy()
        self.H_2 = self.H_2_true.copy()
        if self.channel_est_error:
            # Generate and apply noise to H_1
            noise_H1 = impulsive_noise_generation_np(
                N=self.H_1.size,
                lambda_=self.impulsive_lambda,
                sigma0=self.impulsive_sigma0,
                sigma1=self.impulsive_sigma1
            ).reshape(self.H_1.shape)
            self.H_1 += noise_H1

        noise_H2 = impulsive_noise_generation_np(
            N=self.H_2.size,
            lambda_=self.impulsive_lambda,
            sigma0=self.impulsive_sigma0,
            sigma1=self.impulsive_sigma1
        ).reshape(self.H_2.shape)
        self.H_2 += noise_H2

        init_action_G = np.hstack((np.real(self.G.reshape(1, -1)), np.imag(self.G.reshape(1, -1))))
        init_action_Phi = np.hstack(
            (np.real(np.diag(self.Phi)).reshape(1, -1), np.imag(np.diag(self.Phi)).reshape(1, -1)))

        init_action = np.hstack((init_action_G, init_action_Phi))

        Phi_real = init_action[:, -2 * self.L:-self.L]
        Phi_imag = init_action[:, -self.L:]

        self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag)

        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        H_2_tilde = self._compute_H_2_tilde()
        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2

        H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)

        self.state = np.hstack((init_action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        return self.state

    def _compute_reward(self, Phi):
        reward = 0
        opt_reward = 0
        impulsive_noise_power = (self.impulsive_sigma0**2) + \
                            (self.impulsive_lambda * self.impulsive_sigma1**2)
        
        for k in range(self.K):
            h_2_k = self.H_2[:, k].reshape(-1, 1)
            g_k = self.G[:, k].reshape(-1, 1)

            x = np.abs(h_2_k.T @ Phi @ self.H_1 @ g_k) ** 2

            x = x.item()

            G_removed = np.delete(self.G, k, axis=1)

            interference = np.sum(np.abs(h_2_k.T @ Phi @ self.H_1 @ G_removed) ** 2)
            #y = interference + (self.K - 1) * self.awgn_var
            y = interference + (self.K - 1) * self.awgn_var + impulsive_noise_power

            rho_k = x / y

            reward += np.log(1 + rho_k) / np.log(2)
            opt_reward += np.log(1 + x / ((self.K - 1) * self.awgn_var)) / np.log(2)

        return reward, opt_reward

    def step(self, action):
        self.episode_t += 1

        action = action.reshape(1, -1)
        if self.channel_est_error:
        # Reset to true channels before applying new noise
            self.H_1 = self.H_1_true.copy()
            self.H_2 = self.H_2_true.copy()
            
            # Generate and apply noise to H_1
            noise_H1 = impulsive_noise_generation_np(
                N=self.H_1.size,
                lambda_=self.impulsive_lambda,
                sigma0=self.impulsive_sigma0,
                sigma1=self.impulsive_sigma1
            ).reshape(self.H_1.shape)
            self.H_1 += noise_H1
            
            # Generate and apply noise to H_2
            noise_H2 = impulsive_noise_generation_np(
                N=self.H_2.size,
                lambda_=self.impulsive_lambda,
                sigma0=self.impulsive_sigma0,
                sigma1=self.impulsive_sigma1
            ).reshape(self.H_2.shape)
            self.H_2 += noise_H2

        G_real = action[:, :self.M ** 2]
        G_imag = action[:, self.M ** 2:2 * self.M ** 2]

        Phi_real = action[:, -2 * self.L:-self.L]
        Phi_imag = action[:, -self.L:]

        self.G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)

        self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag)

        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        H_2_tilde = self._compute_H_2_tilde()

        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2

        H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)

        self.state = np.hstack((action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        reward, opt_reward = self._compute_reward(self.Phi)

        done = opt_reward == reward

        return self.state, reward, done, None

    def close(self):
        pass
