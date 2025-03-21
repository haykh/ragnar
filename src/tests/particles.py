import ragnar as rg
import numpy as np

rg.Initialize()


def test_prtls():
    prtls = rg.Particles_3D("electrons")

    assert prtls.label() == "electrons", "Label is not correct"
    assert not prtls.is_allocated(), "Particles are allocated"

    rng = np.random.default_rng(123)

    U1s = rng.random(10)
    U3s = rng.random(10)
    E1s = rng.random(10)
    E2s = rng.random(10)
    E3s = rng.random(10)
    B1s = rng.random(10)
    B2s = rng.random(10)
    B3s = rng.random(10)
    prtls.fromArrays(
        {
            "U1": U1s,
            "U3": U3s,
            "E1": E1s,
            "E2": E2s,
            "E3": E3s,
            "B1": B1s,
            "B2": B2s,
            "B3": B3s,
        }
    )
    for q, f in zip(
        [U1s, U3s, E1s, E2s, E3s, B1s, B2s, B3s],
        [
            prtls.U(1),
            prtls.U(3),
            prtls.E(1),
            prtls.E(2),
            prtls.E(3),
            prtls.B(1),
            prtls.B(2),
            prtls.B(3),
        ],
    ):
        assert np.all(np.isclose(q, f.as_array())), "Arrays are not equal"

    assert prtls.is_allocated(), "Particles are not allocated"
    assert prtls.nactive() == 10, "Number of active particles is not correct"
    assert prtls.nalloc() == 10, "Alocated number of active particles is not correct"

    U1News = rng.random(10)
    U3News = rng.random(10)
    E1News = rng.random(10)
    E2News = rng.random(10)
    E3News = rng.random(10)
    B1News = rng.random(10)
    B2News = rng.random(10)
    B3News = rng.random(10)
    prtls.fromArrays(
        {
            "U1": U1News,
            "U3": U3News,
            "E1": E1News,
            "E2": E2News,
            "E3": E3News,
            "B1": B1News,
            "B2": B2News,
            "B3": B3News,
        },
        append=True,
    )
    for q, qnew, f in zip(
        [U1s, U3s, E1s, E2s, E3s, B1s, B2s, B3s],
        [U1News, U3News, E1News, E2News, E3News, B1News, B2News, B3News],
        [
            prtls.U(1),
            prtls.U(3),
            prtls.E(1),
            prtls.E(2),
            prtls.E(3),
            prtls.B(1),
            prtls.B(2),
            prtls.B(3),
        ],
    ):
        assert np.all(np.isclose(q, f.as_array()[:10])), "Original arrays are not equal"
        assert np.all(
            np.isclose(qnew, f.as_array()[10:])
        ), "Appended arrays are not equal"

    assert prtls.nactive() == 20, "Number of active particles is not correct"
    assert prtls.nalloc() == 20, "Alocated number of active particles is not correct"


def test_prtls_preallocated():
    rng = np.random.default_rng(123)

    prtls = rg.Particles_3D("electrons")
    prtls.allocate(20)

    assert prtls.nactive() == 0, "Number of active particles is not correct"
    assert prtls.nalloc() == 20, "Alocated number of active particles is not correct"

    U1s = rng.random(10)
    U3s = rng.random(10)
    E1s = rng.random(10)
    E2s = rng.random(10)
    E3s = rng.random(10)
    B1s = rng.random(10)
    B2s = rng.random(10)
    B3s = rng.random(10)
    prtls.fromArrays(
        {
            "U1": U1s,
            "U3": U3s,
            "E1": E1s,
            "E2": E2s,
            "E3": E3s,
            "B1": B1s,
            "B2": B2s,
            "B3": B3s,
        },
        append=True,
    )

    assert prtls.nactive() == 10, "Number of active particles is not correct"
    assert prtls.nalloc() == 20, "Alocated number of active particles is not correct"

    for q, f in zip(
        [U1s, U3s, E1s, E2s, E3s, B1s, B2s, B3s],
        [
            prtls.U(1),
            prtls.U(3),
            prtls.E(1),
            prtls.E(2),
            prtls.E(3),
            prtls.B(1),
            prtls.B(2),
            prtls.B(3),
        ],
    ):
        assert np.all(np.isclose(q, f.as_array())), "Arrays are not equal"

    U1News = rng.random(10)
    U3News = rng.random(10)
    E1News = rng.random(10)
    E2News = rng.random(10)
    E3News = rng.random(10)
    B1News = rng.random(10)
    B2News = rng.random(10)
    B3News = rng.random(10)
    prtls.fromArrays(
        {
            "U1": U1News,
            "U3": U3News,
            "E1": E1News,
            "E2": E2News,
            "E3": E3News,
            "B1": B1News,
            "B2": B2News,
            "B3": B3News,
        },
        append=True,
    )

    assert prtls.nactive() == 20, "Number of active particles is not correct"
    assert prtls.nalloc() == 20, "Alocated number of active particles is not correct"

    for q, qnew, f in zip(
        [U1s, U3s, E1s, E2s, E3s, B1s, B2s, B3s],
        [U1News, U3News, E1News, E2News, E3News, B1News, B2News, B3News],
        [
            prtls.U(1),
            prtls.U(3),
            prtls.E(1),
            prtls.E(2),
            prtls.E(3),
            prtls.B(1),
            prtls.B(2),
            prtls.B(3),
        ],
    ):
        assert np.all(np.isclose(q, f.as_array()[:10])), "Original arrays are not equal"
        assert np.all(
            np.isclose(qnew, f.as_array()[10:])
        ), "Appended arrays are not equal"


def test_prtls_coords():
    rng = np.random.default_rng(123)
    prtls = rg.Particles_3D("electrons")
    X1s = rng.random(10)
    U3s = rng.random(10)
    prtls.fromArrays(
        {
            "X1": X1s,
            "U3": U3s,
        },
        append=True,
    )
    assert np.all(np.isclose(prtls.X(1).as_array(), X1s)), "Arrays are not equal"
    assert np.all(np.isclose(prtls.U(3).as_array(), U3s)), "Arrays are not equal"
    assert np.all(
        np.isclose(prtls.U(1).as_array(), np.zeros_like(U3s))
    ), "Arrays are not equal"
    assert np.all(
        np.isclose(prtls.E(1).as_array(), np.zeros_like(U3s))
    ), "Arrays are not equal"
