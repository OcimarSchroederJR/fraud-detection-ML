from pathlib import Path

from src.ingestion.load_data import PROJECT_ROOT, load_raw_transactions


def test_load_raw_transactions_resolves_relative_path_from_project_root(tmp_path, monkeypatch):
    """Caminhos relativos em config.yaml devem ser resolvidos a partir da
    raiz do projeto, não do diretório de trabalho atual — sem isso, o
    carregamento quebra ao rodar a partir de notebooks/ (cwd diferente
    da raiz), como aconteceu de fato ao executar o notebook de EDA."""
    relative_raw = Path("data") / "raw" / "_test_ingestion_sample.csv"
    absolute_raw = PROJECT_ROOT / relative_raw
    absolute_raw.parent.mkdir(parents=True, exist_ok=True)
    absolute_raw.write_text("Time,Amount,Class\n0,1.0,0\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'paths:\n  raw_data: "{relative_raw.as_posix()}"\n', encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    try:
        df = load_raw_transactions(config_path)
        assert len(df) == 1
    finally:
        absolute_raw.unlink()
