import numpy as np

def generateRandomDirection():
    Z = np.random.random() * 2 - 1
    TH = np.random.random() * 2 * np.pi
    return np.array([np.sqrt(1 - Z**2) * np.cos(TH), np.sqrt(1 - Z**2) * np.sin(TH), Z])


def Jonesth(Gamma, x):
    q = x / (1 + Gamma * (1 - x))
    return 2 * q * np.log(q) + (1 + 2 * q) * (1 - q) + 0.5 * Gamma**2 * q**2 * (1 - q) / (1 + Gamma * q)


def sigmaCompton(e_photon):
    """
    Compton cross section (normalized) as a function of photon energy (rest frame of the particle).

    Args:
      e_photon                photon energy

    Returns:
      sigma                   normalized cross section
    """
    sigma = ((1.0 + e_photon) / e_photon**3) *\
        (2.0 * e_photon * (1.0 + e_photon) / (1.0 + 2.0 * e_photon) - np.log(1.0 + 2.0 * e_photon)) +\
        (np.log(1.0 + 2.0 * e_photon) / (2.0 * e_photon)) -\
        (1.0 + 3.0 * e_photon) / (1.0 + 2.0 * e_photon)**2
    sigma /= 1.4
    return sigma


def sigmaDiffCompton(mu, e_photon):
    """
    Compton differential cross section (normalized) as a function of photon energy and cos(theta) = mu (in the rest frame of the particle).

    Args:
      mu                      cos(theta)
      e_photon:               photon energy

    Returns:
      sigmaDiff               normalized differential cross section
    """
    sigmaDiff = 3.0 * (1.0 + mu**2) * (1.0 + (e_photon**2 * (1.0 - mu)**2) /
                                       ((1.0 + e_photon * (1.0 - mu)) * (1.0 + mu**2))) /\
        (16.0 * np.pi * (1.0 + e_photon * (1.0 - mu))**2)
    return sigmaDiff


def getMuScat(e_photon):
    """
    Generate random scattering angle (in the rest frame of the particle).

    Args:
      e_photon:               photon energy

    Returns:
      x1                      Cosine of scattering angle
    """
    x2 = 1000
    x1 = 100
    sig = 0
    while x2 > sig:
        x1 = (np.random.random() - 0.5) * 2
        x2 = np.random.random() * 0.15
        sig = sigmaDiffCompton(x1, e_photon)
    return x1


# vectorized version of the function
getMuScat_vec = np.vectorize(getMuScat)


def Lorentz_transform(p_particle, P_photons):
    """
    Vectorized Lorentz transformation of photon momenta to the frame of a particle.

    Args:
      p_particle:             particle 4-momentum (shape = 4)
      P_photons:              array of photon 4-momenta (shape = 4 x N)

    Returns:
      P_photons_prime         array of photon 4-momenta in the frame of the particle (shape = 4 x N)
    """
    gamma = p_particle[0]
    E_photons = P_photons[:, 0]
    u_particle_unit = p_particle[1:] / np.linalg.norm(p_particle[1:])
    Uph_dot_uprtl = P_photons[:, 1:].dot(u_particle_unit)
    uprtl_mult_Dot = np.array([u_particle_unit]) * np.array([Uph_dot_uprtl]).T
    uprtl_mult_Eph = np.array([p_particle[1:]]) * np.array([E_photons]).T
    U_photons_prime = P_photons[:, 1:] + \
        (gamma - 1.0) * uprtl_mult_Dot - uprtl_mult_Eph
    E_photons_prime = np.linalg.norm(U_photons_prime, axis=1)
    P_photons_prime = np.insert(U_photons_prime, 0, E_photons_prime, axis=1)
    return P_photons_prime

# Main function


def ComptonScatter(p_particle, P_photons):
    """
    Vectorized scattering function

    Args:
      p_particle:             particle 4-momentum (shape = 4)
      P_photons:              array of photon 4-momenta (shape = 4 x N)

    Returns:
      E_photons_2             array of photon energies after the scattering (shape = N)
    """

    P_photons_1 = Lorentz_transform(p_particle, P_photons)

    # compute interaction probabilities
    probabilities = sigmaCompton(
        P_photons_1[:, 0]) * P_photons_1[:, 0] / (p_particle[0] * P_photons[:, 0])
    # normalize interaction probabilities
    print(probabilities.max())
    probabilities /= probabilities.max()

    p_ = np.random.random(len(probabilities))

    # select only photons that are scattered
    P_photons_1_scat = P_photons_1[p_ < probabilities]

    # generate random scattering angles
    mu_scat = getMuScat_vec(P_photons_1_scat[:, 0])

    a_basis = P_photons_1_scat[:, 1:] / P_photons_1_scat[:, 0, None]
    b_ = np.array([generateRandomDirection() for i in range(len(a_basis))])
    b_basis = np.cross(a_basis, b_, axis=1)
    b_basis /= np.linalg.norm(b_basis, axis=1)[:, None]

    # photon energy after scattering in the electron frame
    E_photons_2_scat = P_photons_1_scat[:, 0] / \
        (1.0 + P_photons_1_scat[:, 0] * (1.0 - mu_scat))
        
    # E_photons_2 = P_photons_1_scat[:, 1]
    # photon momentum after scattering in the electron frame
    U_photons_2_scat = E_photons_2_scat[:, None] * (
        mu_scat[:, None] * a_basis + np.sqrt(1.0 - mu_scat**2)[:, None] * b_basis)

    P_photons_2_scat = np.insert(U_photons_2_scat, 0, E_photons_2_scat, axis=1)

    # transforming back to lab frame (in p_particle only reverse momenta)
    E_photons_2 = Lorentz_transform(
        np.array([1.0, -1.0, -1.0, -1.0]) * p_particle, P_photons_2_scat)[:, 0]
    # return photon energy after scattering in lab frame
    return E_photons_2
