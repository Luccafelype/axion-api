from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from scipy.optimize import linprog

app = FastAPI()

class Ingrediente(BaseModel):
    nome: str
    proteina: float
    carbo: float
    gordura: float
    kcal: float

class CalculoRequest(BaseModel):
    metas: dict
    ingredientes: List[Ingrediente]

@app.post("/calcular-prato")
def calcular_prato(payload: CalculoRequest):
    metas = payload.metas
    ingredientes = payload.ingredientes

    # Matrizes
    proteinas = [i.proteina / 100 for i in ingredientes]
    carbos = [i.carbo / 100 for i in ingredientes]
    gorduras = [i.gordura / 100 for i in ingredientes]
    kcals = [i.kcal / 100 for i in ingredientes]

    A = [
        [-p for p in proteinas],
        [-c for c in carbos],
        [g for g in gorduras],
        [k for k in kcals]
    ]
    b = [
        -metas["proteina"],
        -metas["carbo"],
        metas["gordura"],
        metas["kcal"]
    ]
    c = [1 for _ in ingredientes]

    result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')

    if result.success:
        porcoes = [
            {"ingrediente": i.nome, "gramas": round(q, 1)}
            for i, q in zip(ingredientes, result.x)
        ]
        return {
            "sucesso": True,
            "porcoes_calculadas": porcoes
        }
    else:
        return {
            "sucesso": False,
            "erro": "Não foi possível calcular uma combinação viável."
        }
