from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class ValidateResult:
    ok: bool
    message: str

class ParamValidator:
    @staticmethod
    def basic(model_name: str, model_path: str, data: str) -> ValidateResult:
        if not (model_name or model_path):
            return ValidateResult(False, "Не выбрана модель: стандартная (из списка) или файл в кэше.")
        if not data:
            return ValidateResult(False, "Не указан путь к данным (data).")
        return ValidateResult(True, "OK")

    @staticmethod
    def require(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(msg)

    @staticmethod
    def range_int(name: str, v: int, lo: int, hi: int) -> int:
        ParamValidator.require(lo <= v <= hi, f"{name} должен быть в диапазоне [{lo}..{hi}]")
        return int(v)
