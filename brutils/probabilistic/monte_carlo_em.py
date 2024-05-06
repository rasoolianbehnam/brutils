import warnings

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import tensorflow.keras as keras
from logging import Logger

logger = Logger("mcem")

ACCEPTANCE_RATIO_THRESHOLD = .1


class MonteCarloEM:
    def __init__(
            self, model,
            trainable_variables=None, target_log_prob_fn=None, *,
            learning_rate=1e-2, n_e_samples=1,
            step_size=5e-3, num_leapfrog_steps=10
    ):
        if trainable_variables is None:
            trainable_variables = model.trainable_variables
        target_log_prob_fn = target_log_prob_fn or model.log_prob
        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn, step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps
        )
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.current_state = model.sample(n_e_samples)
        self.kernel_results = hmc.bootstrap_results(self.current_state)
        self.learning_rate = learning_rate

        @tf.function(autograph=False, jit_compile=True)
        def one_e_step(current_state, kernel_results):
            return hmc.one_step(current_state, kernel_results)

        @tf.function(autograph=False, jit_compile=True)
        def k_e_steps(k, current_state, kernel_results, num_accepted=tf.zeros([n_e_samples])):
            return tf.while_loop(
                lambda i, *_: i < k,
                lambda i, state, kr, num_accepted: (
                    i + 1, *one_e_step(state, kr), num_accepted + tf.cast(kr.is_accepted, tf.float32)),
                (0, current_state, kernel_results, num_accepted)
            )[1:]

        @tf.function(autograph=False, jit_compile=True)
        def one_m_step(current_state):
            with tf.GradientTape() as tape:
                loss = -tf.reduce_mean(target_log_prob_fn(current_state))  # / len(train)
            grads = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            return loss

        self.one_e_step = one_e_step
        self.k_e_steps = k_e_steps
        self.one_m_step = one_m_step

        self.num_accepted = np.zeros((n_e_samples,))
        self.loss_history = []

    def warmup(self, num_warmup_iters):
        if num_warmup_iters == 0:
            return self.current_state, self.kernel_results
        self.current_state, self.kernel_results, n_acc = self.k_e_steps(
            num_warmup_iters,
            self.current_state,
            self.kernel_results
        )
        acc_ratio = n_acc.numpy().mean() / num_warmup_iters
        print(f"Acceptance Rate after {num_warmup_iters:05d} Steps: {acc_ratio:4.2f}")

    def train(self, num_iters, num_warmup_iters, n_e_steps_per_iter=5, print_every=None):
        if print_every is None:
            print_every = int(1 / self.learning_rate)
        print(f"*** Starting warmup for {num_warmup_iters} Iterations")
        self.warmup(num_warmup_iters)

        print(f"*** Starting EM for {num_iters} Iterations")
        for t in range(num_iters + 1):
            self.current_state, self.kernel_results, *tmp = self.k_e_steps(
                n_e_steps_per_iter, self.current_state, self.kernel_results
            )
            loss = self.one_m_step(self.current_state)
            self.num_accepted += self.kernel_results.is_accepted.numpy()
            self.loss_history.append(loss)
            if t % print_every == 0:
                acc_ratios = self.num_accepted / len(self.loss_history) / n_e_steps_per_iter
                acc_ratio_mean = acc_ratios.mean()
                print(
                    f"Epoch {len(self.loss_history):05d} | "
                    f"Acceptance Rate: {acc_ratio_mean:4.2f}/{acc_ratios.min():4.2f} | Loss {loss:10.5f}"
                )
                if acc_ratio_mean() < ACCEPTANCE_RATIO_THRESHOLD:
                    logger.warning(
                        f"Acceptance ration below threshold {acc_ratio_mean:4.2f}/{ACCEPTANCE_RATIO_THRESHOLD:4.2f}"
                    )
                look_back = int(1 / self.learning_rate)
                tmp = self.loss_history[-look_back:]
                if len(tmp) > 10 and np.polyfit(np.arange(len(tmp)), tmp, 1)[0] > -self.learning_rate:
                    print("Convergence Reached")
                    break
