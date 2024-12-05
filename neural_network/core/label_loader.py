import json

class LabelLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.labels = self._load_labels()

    def _load_labels(self) -> dict:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Erro ao carregar o arquivo {self.file_path}: {e}")

    def get_label(self, key: int) -> str:
        return self.labels.get(str(key), "Desconhecido")