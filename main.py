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

    # Função objetivo: penaliza proporção fora do habitual, pouco uso de ingrediente e favorece volume se solicitado
    def objetivo(x):
        prop_penalty = 0
        for i in range(n):
            for j in range(i+1, n):
                r_ideal = ref_pesos[i] / ref_pesos[j]
                r_real = (x[i] + 1e-6) / (x[j] + 1e-6)
                desv = (r_real - r_ideal) / r_ideal
                if abs(desv) > 0.2:
                    prop_penalty += desv**2
        uso_penalty = 0
        for i, ingr in enumerate(ingredientes):
            min_uso = 5 if ingr.gordura > 90 else 30
            if x[i] < min_uso:
                uso_penalty += ((min_uso - x[i]) / min_uso) ** 2
        volume_penalty = -np.sum(x) / 1000 if maximizar_volume else 0
        return prop_penalty + 0.2 * uso_penalty + volume_penalty

    bounds = []
    for ingr, ref in zip(ingredientes, ref_pesos):
        min_uso = 5 if ingr.gordura > 90 else min(30, ref*0.5)
        bounds.append( (min_uso, None) )
    x0 = ref_pesos.copy()

    constraints = [
        # Proteína ±5%
        {"type": "ineq", "fun": lambda x: np.dot(proteina, x) - metas['proteina'] * 0.95},
        {"type": "ineq", "fun": lambda x: metas['proteina'] * 1.05 - np.dot(proteina, x)},
        # Carboidrato: -10% até +20%
        {"type": "ineq", "fun": lambda x: np.dot(carbo, x) - metas['carbo'] * 0.90},
        {"type": "ineq", "fun": lambda x: metas['carbo'] * 1.20 - np.dot(carbo, x)},
        # Gordura: -50% até a meta (nunca acima)
        {"type": "ineq", "fun": lambda x: np.dot(gordura, x) - metas['gordura'] * 0.50},
        {"type": "ineq", "fun": lambda x: metas['gordura'] - np.dot(gordura, x)},
        # Calorias ±2%
        {"type": "ineq", "fun": lambda x: np.dot(kcal, x) - metas['kcal'] * 0.98},
        {"type": "ineq", "fun": lambda x: metas['kcal'] * 1.02 - np.dot(kcal, x)},
    ]

    res = minimize(objetivo, x0=x0, bounds=bounds, constraints=constraints, method='SLSQP')

    if res.success:
        porcoes = [
            {"ingrediente": i.nome, "gramas": max(0, int(round(q)))}
            for i, q in zip(ingredientes, res.x)
        ]
        return {
            "sucesso": True,
            "porcoes_calculadas": porcoes
        }
    else:
        # Diagnóstico inteligente: verifica quais macros ficaram fora da faixa
        tentativa = res.x if res.x is not None else x0
        prot = np.dot(proteina, tentativa)
        carb = np.dot(carbo, tentativa)
        gord = np.dot(gordura, tentativa)
        kc = np.dot(kcal, tentativa)
        problemas = []
        # Proteína
        if prot < metas['proteina'] * 0.95:
            problemas.append("proteína baixa")
        if prot > metas['proteina'] * 1.05:
            problemas.append("proteína alta")
        # Carbo
        if carb < metas['carbo'] * 0.90:
            problemas.append("carboidrato baixo")
        if carb > metas['carbo'] * 1.20:
            problemas.append("carboidrato alto")
        # Gordura
        if gord < metas['gordura'] * 0.50:
            problemas.append("gordura baixa")
        if gord > metas['gordura']:
            problemas.append("gordura alta")
        # Kcal
        if kc < metas['kcal'] * 0.98:
            problemas.append("calorias baixas")
        if kc > metas['kcal'] * 1.02:
            problemas.append("calorias altas")
        return {
            "sucesso": False,
            "erro": "Não foi possível calcular uma combinação viável dentro das restrições.",
            "diagnostico": problemas
        }
