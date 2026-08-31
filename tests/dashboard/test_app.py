from streamlit.testing.v1 import AppTest


def test_dashboard_runs_without_model_or_dataset():
    """Fumaça: o dashboard não deve quebrar quando não há modelo treinado
    nem dataset baixado (o caso normal de um checkout novo/CI), só
    mostrar os avisos apropriados."""
    at = AppTest.from_file("dashboard/app.py", default_timeout=30)
    at.run()

    assert not at.exception
