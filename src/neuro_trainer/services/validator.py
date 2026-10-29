from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class ValidateResult:
    ok: bool
    message: str


class ParamValidator:
    @staticmethod
    def basic(model_name: str, model_path: str, data: str, need_data: bool = True) -> ValidateResult:
        """ Минимальная проверка перед запуском: выбрана модель и указан датасет. """
        if not (model_name or model_path):
            return ValidateResult(False, "Не выбрана модель: стандартная из списка или свой файл .pt.")
        if need_data and not data:
            return ValidateResult(False, "Не указан датасет (data.yaml или папка).")
        return ValidateResult(True, "OK")

    @staticmethod
    def require(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(msg)
