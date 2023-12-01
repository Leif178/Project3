import numpy as np
import tqdm
import numpy.random
from matplotlib import pyplot as plt

j0p = 1
j0m = .95
j1p = 2
j1m = 1.4
N = 1000000


def post(x, y):
    return np.exp(-2 * j0p * (1 - x**2) - j0m * (1 - y**2)) / (
        np.exp(-2 * j0p * (1 - x**2) - 2 * j0m * (1 - y**2))
        + np.exp(-j1p * (x**2) - j1m * (y**2))
    )


def proposal(x, y):
    return [np.random.uniform(low=-1), np.random.uniform(low=-1)]


def mcmc(initial, post, prop, iterations):
    x = [initial]
    p = [post(x[-1][0], x[-1][1])]
    for i in tqdm.tqdm(range(iterations)):
        x_test = prop(x[-1][0], x[-1][1])
        p_test = post(x_test[0], x_test[1])

        acc = p_test
        u = np.random.uniform(0, 1)
        if u <= acc and abs(x_test[0]) <= 1 and abs(x_test[1]) <= 1:
            x.append(x_test)
            p.append(p_test)
    return x, p


chain, prob = mcmc([0, 0], post, proposal, N)


def requiv(chain):
    """


    Parameters
    ----------
    chain : TYPE
        DESCRIPTION.

    Returns
    -------
    chainx : TYPE
        DESCRIPTION.
    chainy : TYPE
        DESCRIPTION.
    chainr : TYPE
        DESCRIPTION.

    """
    chainx = []
    chainy = []
    chainr = []
    for item in chain:
        chainx.append(item[0])
        chainy.append(item[1])
        chainr.append(np.sqrt(item[0] ** 2 + item[1] ** 2))
    return chainx, chainy, chainr


chainx, chainy, chainr = requiv(chain)

plt.figure()
plt.title("Evolution of the walker")
plt.plot(chainr)
plt.ylabel("reff-value")
plt.xlabel("Iteration")
"""
plt.figure()
plt.title("Evolution of the walker")
plt.plot(chainy)
plt.ylabel('y-value')
plt.xlabel('Iteration')
"""


plt.figure()
plt.title("Evolution of the walker")
plt.plot(chainr)
plt.xlim(0, 100)
plt.ylabel("reff-value")
plt.xlabel("Iteration")

plt.figure()
plt.hist2d(chainx, chainy, bins=80)
plt.xlim(-1, 1)
plt.ylim(-1, 1)

"""
Gelman Rubin Statistic
"""

icsx = np.arange(-1, 1, 0.5)
icsy = np.arange(-1, 1, 0.5)


chainmeanx = []
chainmeany = []
chainmeanr = []

chainvarx = []
chainvary = []
chainvarr = []

for x in icsx:
    for y in icsy:
        chain, prob = mcmc([x, y], post, proposal, N)
        chainx, chainy, chainr = requiv(chain)
        D = len(chainx) // 2
        L = len(chainx) - D
        chainmeanx.append(1 / L * sum(chainx[D:]))
        chainmeany.append(1 / L * sum(chainy[D:]))
        chainmeanr.append(1 / L * sum(chainr[D:]))

        chainvarx.append(
            1 / (L - 1) * sum((np.array(chainx[D:]) - chainmeanx[-1]) ** 2)
        )
        chainvary.append(
            1 / (L - 1) * sum((np.array(chainy[D:]) - chainmeany[-1]) ** 2)
        )
        chainvarr.append(
            1 / (L - 1) * sum((np.array(chainr[D:]) - chainmeanr[-1]) ** 2)
        )

grandmeanx = 1 / len(chainmeanx) * sum(chainmeanx)
grandmeany = 1 / len(chainmeany) * sum(chainmeany)
grandmeanr = 1 / len(chainmeanr) * sum(chainmeanr)

Bx = L / (len(chainmeanx) - 1) * np.sum((np.array(chainmeanx) - grandmeanx) ** 2)
By = L / (len(chainmeany) - 1) * np.sum((np.array(chainmeany) - grandmeany) ** 2)
Br = L / (len(chainmeanr) - 1) * np.sum((np.array(chainmeanr) - grandmeanr) ** 2)

Wx = 1 / len(chainmeanx) * np.sum((np.array(chainvarx)) ** 2)
Wy = 1 / len(chainmeany) * np.sum((np.array(chainvary)) ** 2)
Wr = 1 / len(chainmeanr) * np.sum((np.array(chainvarr)) ** 2)


Rx = ((L - 1) / L * Wx - 1 / L * Bx) / Wx
Ry = ((L - 1) / L * Wy - 1 / L * By) / Wy
Rr = ((L - 1) / L * Wr - 1 / L * Br) / Wr
print("Gelman Rubin Statistics:")
print("x : ", Rx, " y : ", Ry, " r : ", Rr)


plt.show()
