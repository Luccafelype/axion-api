
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from scipy.optimize import minimize
import numpy as np

app = FastAPI()

class Ingrediente(BaseModel):
    nome: str
    proteina: float
    carbo: float
    gordura: float
    kcal: float
    peso_referencia: Optional[float] = None

class CalculoRequest(BaseModel):
    metas: dict
    ingredientes: List[Ingrediente]
    maximizar_volume: Optional[bool] = False

@app.post("/calcular-prato")
def calcular_prato(payload: CalculoRequest):
    metas = payload.metas
    ingredientes = payload.ingredientes
    maximizar_volume = payload.maximizar_volume

    n = len(ingredientes)
    proteina = np.array([i.proteina / 100 for i in ingredientes])
    carbo = np.array([i.carbo / 100 for i in ingredientes])
    gordura = np.array([i.gordura / 100 for i in ingredientes])
    kcal = np.array([i.kcal / 100 for i in ingredientes])
    ref_pesos = np.array([i.peso_referencia or 100 for i in ingredientes])

    # Objetivo: minimizar erro total em relação às metas + penalidades
    def objetivo(x):
        p_total = np.dot(proteina, x)
        c_total = np.dot(carbo, x)
        g_total = np.dot(gordura, x)
        kcal_total = np.dot(kcal, x)

        erro_p = ((p_total - metas["proteina"]) / metas["proteina"]) ** 2
        erro_c = ((c_total - metas["carbo"]) / metas["carbo"]) ** 2
        erro_g = ((g_total - metas["gordura"]) / metas["gordura"]) ** 2
        erro_k = ((kcal_total - metas["kcal"]) / metas["kcal"]) ** 2

        erro_macro = erro_p + erro_c + erro_g + erro_k

        # Penalidade por quebrar proporção
        prop_penalty = 0
        for i in range(n):
            for j in range(i+1, n):
                r_ideal = ref_pesos[i] / ref_pesos[j]
                r_real = (x[i] + 1e-6) / (x[j] + 1e-6)
                desv = (r_real - r_ideal) / r_ideal
                if abs(desv) > 0.2:  # mais de 20% fora
                    prop_penalty += desv**2

        # Penalidade por não usar algum ingrediente (>10g mínimo exceto gordura pura)
        uso_penalty = 0
        for i, ingr in enumerate(ingredientes):
            min_uso = 5 if ingr.gordura > 90 else 30
            if x[i] < min_uso:
                uso_penalty += ((min_uso - x[i]) / min_uso) ** 2

        # Penalidade por volume baixo se maximizar_volume = True
        volume = np.sum(x)
        volume_penalty = -volume / 1000 if maximizar_volume else 0

        return erro_macro + 0.5 * prop_penalty + 0.2 * uso_penalty + volume_penalty

    bounds = [(0, None) for _ in range(n)]
    x0 = ref_pesos.copy()

    res = minimize(objetivo, x0=x0, bounds=bounds, method='SLSQP')

    if res.success:
        porcoes = [
            {"ingrediente": i.nome, "gramas": round(q, 1)}
            for i, q in zip(ingredientes, res.x)
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
