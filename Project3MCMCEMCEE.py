import numpy as np
import tqdm
import emcee
from matplotlib import pyplot as plt

j0p = 15
j0m = 3
j1p = 75
j1m = 100

def post(x, y):
    log_term1 = -2 * j0p * (1 - x**2) - 2 * j0m * (1 - y**2)
    log_term2 = -j1p * (x**2) - j1m * (y**2)
    return np.exp(log_term1 - np.logaddexp(log_term1, log_term2))

def emcee_post(params):
    x, y = params
    log_term1 = -2 * j0p * (1 - x**2) - 2 * j0m * (1 - y**2)
    log_term2 = -j1p * (x**2) - j1m * (y**2)
    return np.exp(log_term1 - np.logaddexp(log_term1, log_term2))

def proposal(x, y):
    return [np.random.uniform(low=-1), np.random.uniform(low=-1)]

def mcmc_with_burn_in(initial, post, prop, iterations, burn_in):
    x = [initial]
    p = [post(x[-1][0], x[-1][1])]

    for i in tqdm.tqdm(range(iterations)):
        x_test = prop(x[-1][0], x[-1][1])
        p_test = post(x_test[0], x_test[1])
        acc = p_test / p[-1]
        u = np.random.uniform(0, 1)

        if u <= acc and abs(x_test[0]) <= 1 and abs(x_test[1]) <= 1:
            x.append(x_test)
            p.append(p_test)

    # Discard the burn-in samples
    x = x[burn_in:]
    p = p[burn_in:]

    return x, p

def emcee_sampling(initial, post, nwalkers, nsteps):
    ndim = 2
    pos = [initial + 1e-4 * np.random.randn(ndim) for _ in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, post)

    # Run the MCMC sampling
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Get the samples
    samples = sampler.get_chain()[:, :, :].reshape((-1, ndim))

    return samples


# Convergence testing methods
def gelman_rubin_statistic(chains):
    """
    Calculate the Gelman-Rubin statistic for convergence testing.
    """
    m, n = chains.shape
    chain_means = np.mean(chains, axis=1)
    between_chain_variance = n / (m - 1) * np.sum((chain_means - np.mean(chains))**2)
    within_chain_variance = 1 / m * np.sum(np.var(chains, axis=1))
    estimated_variance = (n - 1) / n * within_chain_variance + 1 / n * between_chain_variance
    r_hat = np.sqrt(estimated_variance / within_chain_variance)
    return r_hat

def plot_trace(chain, param_names=None, line_style='-', color=None):
    """
    Plot trace plots for each parameter in the chain.
    """
    n_params = chain.shape[1]
    plt.figure(figsize=(12, 8))
    for i in range(n_params):
        plt.subplot(n_params, 1, i + 1)
        plt.plot(chain[:, i], label=f'Parameter {i + 1}', linestyle=line_style, color=color)
        plt.xlabel('Iteration')
        plt.ylabel(f'Parameter {i + 1} value')
        plt.legend()
    plt.tight_layout()
    plt.show()

# Identify Stable Regions
def identify_stable_regions(trace, threshold=0.02):
    """
    Identify stable regions in the trace plot.
    """
    stable_start = 0
    stable_end = 0
    stable_regions = []

    for i in range(1, len(trace)):
        if abs(trace[i] - trace[i-1]) < threshold:
            if stable_start == 0:
                stable_start = i - 1
        else:
            if stable_start > 0:
                stable_end = i
                stable_regions.append((stable_start, stable_end))
                stable_start = 0

    return stable_regions

# Burn-in phase for MCMC
burn_in_iterations = 5000  # Adjust the burn-in iterations as needed
burn_in_samples, burn_in_probs = mcmc_with_burn_in([1, 1], post, proposal, iterations=1000000 + burn_in_iterations, burn_in=burn_in_iterations)

# Extract the samples after burn-in
chain, prob = burn_in_samples, burn_in_probs

# Convert the list to a NumPy array
chain = np.array(chain)

# Gelman-Rubin Statistic
r_hat = gelman_rubin_statistic(chain)
print("Gelman-Rubin Statistic:", r_hat)

# Plotting trace plots with line style '-.' and color 'purple'
plot_trace(chain, line_style='-.', color='purple')

# Emcee
emcee_samples = emcee_sampling([1, 1], emcee_post, nwalkers=100, nsteps=1000)

# Function to calculate autocorrelation lengths with fixed seed
def calculate_autocorr_lengths_fixed_seed(samples, n_runs=5, random_seed=42):
    autocorr_lengths_list = []

    for _ in range(n_runs):
        np.random.seed(random_seed)
        emcee_samples = emcee_sampling([1, 1], emcee_post, nwalkers=100, nsteps=1000)
        
        autocorr_lengths = np.zeros(emcee_samples.shape[1])
        for i in range(emcee_samples.shape[1]):
            autocorr_lengths[i] = emcee.autocorr.integrated_time(emcee_samples[:, i])

        autocorr_lengths_list.append(autocorr_lengths)

    # Calculate the average autocorrelation lengths
    avg_autocorr_lengths = np.mean(autocorr_lengths_list, axis=0)

    return avg_autocorr_lengths

# Calculate and print average autocorrelation lengths with fixed seed
avg_autocorr_lengths_fixed_seed = calculate_autocorr_lengths_fixed_seed(emcee_samples)
print("Average Autocorrelation Lengths (Fixed Seed):", avg_autocorr_lengths_fixed_seed)


# Plotting
plt.figure()
plt.title("Evolution of the walker (Metropolis-Hastings)")
plt.plot(np.sqrt(np.array(chain)[:, 0]**2 + np.array(chain)[:, 1]**2), label='Metropolis-Hastings')
plt.xlabel('Iteration')
plt.ylabel('reff-value')
plt.legend()

plt.figure()
plt.title("Evolution of the walker")
plt.plot(np.sqrt(np.array(chain)[:, 0]**2 + np.array(chain)[:, 1]**2))
plt.xlim(0, 50)
plt.ylabel('reff-value')
plt.xlabel('Iteration')

plt.figure()
plt.hist2d(np.array(chain)[:, 0], np.array(chain)[:, 1], bins=100)
plt.xlim(-1, 1)
plt.ylim(-1, 1)

# Plotting Evolution of the walker (emcee) with Stable Regions
plt.figure()
plt.title("Evolution of the walker (emcee)")
trace_emcee = np.sqrt(emcee_samples[:, 0]**2 + emcee_samples[:, 1]**2)
plt.plot(trace_emcee, color='red', alpha=1.0)
plt.xlabel('Iteration')
plt.ylabel('reff-value')

# Highlight Stable Regions
emcee_stable_regions = identify_stable_regions(trace_emcee)
for start, end in emcee_stable_regions:
    plt.axvspan(start, end, color='green', alpha=0.2)

# Plotting Autocorrelation Length
plt.figure()
plt.bar(range(len(avg_autocorr_lengths_fixed_seed)), avg_autocorr_lengths_fixed_seed, tick_label=[f'Parameter {i+1}' for i in range(len(avg_autocorr_lengths_fixed_seed))])
plt.title('Autocorrelation Lengths for Each Parameter')
plt.xlabel('Parameter')
plt.ylabel('Autocorrelation Length')
plt.show()

# Gelman Rubin Statistic
D = int(len(chain) / 2)
L = len(chain) - D
xbarx = 1 / L * sum(np.array(chain)[D:, 0])
xbary = 1 / L * sum(np.array(chain)[D:, 1])
grandmean = 1 / 2 * (xbarx + xbary)
B = L * ((xbarx - grandmean)**2 + (xbary - grandmean)**2)
inchainvarx = 1 / (L - 1) * sum((np.array(chain)[D:, 0] - xbarx)**2)
inchainvary = 1 / (L - 1) * sum((np.array(chain)[D:, 1] - xbary)**2)
W = 1 / 2 * (inchainvary + inchainvarx)
R = ((L - 1) / L * W - 1 / L * B) / W
print("Gelman Rubin Statistic:", R)

plt.show()
