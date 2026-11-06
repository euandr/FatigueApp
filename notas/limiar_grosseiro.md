# Por que o limiar do detector é "grosseiro"

## Resumo em uma frase

O detector **conta frames, não tempo**. Como frames não têm duração fixa, o
resultado depende do FPS e a contagem de piscadas é só uma aproximação.

---

## O que o código faz

```python
if eyes_closed:
    blink_counter += 1
    if blink_counter >= 3:      # 3 frames com olho fechado
        total_blinks += 1
        blink_counter = 0
```

Ou seja: **conta 1 piscada quando o olho aparece fechado em 3 frames seguidos**.

## O problema

A 10 FPS, 3 frames ≈ **300 ms**.

- Piscada curta (~150 ms) → pode aparecer em 1 frame só → **não conta**;
- Piscada que cai entre duas capturas → **não conta**;
- Só conta se o olho estiver fechado em 3 capturas seguidas.

Pior: se o olho ficar fechado por 30 frames (3 s), o contador soma 1 piscada a
cada 3 frames:

```
30 frames fechados ÷ 3 = 10 piscadas contadas
```

Um único fechamento longo vira **10 piscadas**.

## Outros limiares também são em frames

| Regra no código | A 10 FPS significa |
| --- | --- |
| `counter >= 30` (fadiga) | ~3 s de olhos fechados |
| `blink_counter >= 3` | ~300 ms fechado = 1 piscada |
| `ear_history` de 30 itens | janela de ~3 s |
| `blinks_in_window >= 5` | 5 quedas de EAR em ~3 s |

## Por que isso importa

Como tudo é contado em frames, **mudar o FPS muda o comportamento**:

| FPS | "3 frames" vira | Efeito |
| --- | --- | --- |
| 30 FPS | 100 ms | conta mais piscadas (e mais falsos positivos) |
| 10 FPS | 300 ms | comportamento atual |
| 5 FPS | 600 ms | quase nenhuma piscada conta |

O algoritmo está, na prática, **calibrado para ~10 FPS**. Não está "errado" —
funciona nessa cadência —, mas não é uma medida precisa e não deve ser usado
com outro FPS sem recalibrar.

## Como melhorar (se um dia for necessário)

1. Guardar o **instante de cada frame** e usar **segundos** nos limiares, em vez
   de números de frames;
2. Fixar o FPS de análise e manter a cadência estável;
3. Contar piscadas pela **transição** olho aberto → fechado → aberto, e não por
   blocos de 3 frames fechados.
