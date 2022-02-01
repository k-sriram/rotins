import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--makeplots",
        action="store_true",
        help="make diagnostic plots for manual inspection.",
    )


@pytest.fixture
def makeplots(request):
    return request.config.getoption("--makeplots")
