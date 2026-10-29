"""
Demo didatica: Observabilidade em ML + Atualizacao online de modelos
====================================================================

HISTORIA: um banco usa um modelo para bloquear transacoes de cartao suspeitas.

  ATO 1 - Dia normal        : tudo funciona.
  ATO 2 - Black Friday      : todo mundo compra mais e mais rapido. Os dados MUDAM
                              (data drift), mas fraude continua igual. O modelo
                              antigo passa a bloquear CLIENTES BONS.
  ATO 3 - Golpistas mudam   : "teste de cartao" (valores minusculos, madrugada,
                              varias tentativas). A propria fraude MUDA
                              (concept drift). O modelo antigo nao reconhece.

Dois modelos recebem exatamente as mesmas transacoes:
  * ESTATICO : treinado uma vez, nunca atualizado ("o que acontece se ninguem cuidar").
  * ONLINE   : aprende com cada lote assim que os rotulos (chargebacks) chegam.

Observabilidade:
  - Taxa de bloqueio ........ NAO precisa de rotulo  (sinal rapido)
  - PSI das features ........ NAO precisa de rotulo  (sinal rapido)
  - Recall de fraudes ....... precisa de rotulo, que chega com ATRASO (sinal lento, mas e a verdade)
  - Prejuizo em R$ .......... a metrica de negocio

Uso:
  python demo_fraude.py                 # ao vivo (para gravar)
  python demo_fraude.py --speed 0.25    # mais lento
  python demo_fraude.py --headless      # sem janela; salva dashboard_fraude.png
"""
import argparse
import os
import sys
import time
from collections import deque

import numpy as np
from sklearn.linear_model import SGDClassifier

# ------------------------------------------------------------------ configuracao
SEED = 11
FEATURES = ["log_valor", "madrugada", "log_dist_km", "tx_na_hora"]
BATCH = 250              # transacoes por hora
N_STEPS = 150            # horas simuladas
BF_START, BF_END = 50, 90        # Black Friday
NEW_FRAUD_AT = 90                # golpistas mudam de tatica
LABEL_DELAY = 4          # chargebacks chegam 4 horas depois
WINDOW = 4
PSI_LIMIT = 0.20
RECALL_SLO = 0.70
BLOCK_RATE_FACTOR = 1.8  # alerta se taxa de bloqueio > 1.8x (ou < 1/1.8x) a normal
FP_COST = 30.0           # R$ de atrito por cliente bom bloqueado
FRAUD_FIXED_COST = 120.0 # R$ de chargeback + operacao por fraude que passou
LR_NORMAL, LR_BOOST, BOOST_STEPS = 0.01, 0.06, 10

ACTS = [
    (0, BF_START, "ATO 1  |  Dia normal",
     "O modelo de fraude foi treinado com dados normais e vai bem:\n"
     "bloqueia ~6% das transacoes e captura quase todas as fraudes.", "#2ecc71"),
    (BF_START, BF_END, "ATO 2  |  BLACK FRIDAY",
     "Todo mundo compra mais, mais caro e mais rapido. As fraudes NAO mudaram,\n"
     "mas o modelo estatico nunca viu esse comportamento e bloqueia clientes bons.", "#f39c12"),
    (NEW_FRAUD_AT, N_STEPS, "ATO 3  |  Golpistas mudam de tatica",
     "Nova fraude: 'teste de cartao' - valores minusculos, madrugada, varias tentativas.\n"
     "O modelo antigo acha que valor baixo = cliente bom e deixa passar.", "#e74c3c"),
]

# ----------------------------------------------------------------- terminal
os.system("")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
RED, YEL, GRN, CYA, DIM, BLD, RST = ("\033[91m", "\033[93m", "\033[92m", "\033[96m",
                                      "\033[2m", "\033[1m", "\033[0m")


def log(msg, color=""):
    print(f"{color}{msg}{RST}", flush=True)


def act_of(step):
    for a in ACTS:
        if a[0] <= step < a[1]:
            return a
    return ACTS[-1]


# ------------------------------------------------------------------- dados
class Bank:
    """Gera transacoes. Colunas: log_valor, madrugada, log_dist_km, tx_na_hora."""

    def __init__(self, rng):
        self.rng = rng

    def legit(self, n, black_friday):
        r = self.rng
        lv = r.normal(4.0 + (0.8 if black_friday else 0.0), 0.8, n)
        mad = r.random(n) < (0.18 if black_friday else 0.05)
        ld = r.normal(1.0, 0.8, n)
        tx = r.poisson(2.6 if black_friday else 1.0, n)
        return np.column_stack([lv, mad, ld, tx])

    def fraud_classic(self, n):
        r = self.rng
        return np.column_stack([r.normal(6.0, 0.8, n), r.random(n) < 0.5,
                                r.normal(3.0, 0.8, n), r.poisson(3.0, n)])

    def fraud_card_testing(self, n):
        r = self.rng
        return np.column_stack([r.normal(2.0, 0.5, n), r.random(n) < 0.85,
                                r.normal(1.2, 0.8, n), r.poisson(5.0, n)])

    def batch(self, step, n=BATCH):
        bf = BF_START <= step < BF_END
        new = step >= NEW_FRAUD_AT
        n_fraud = self.rng.binomial(n, 0.08 if new else 0.06)
        Xl = self.legit(n - n_fraud, bf)
        Xf = self.fraud_card_testing(n_fraud) if new else self.fraud_classic(n_fraud)
        X = np.vstack([Xl, Xf]).astype(float)
        y = np.concatenate([np.zeros(n - n_fraud), np.ones(n_fraud)]).astype(int)
        idx = self.rng.permutation(n)
        return X[idx], y[idx]


# ---------------------------------------------------------- observabilidade
def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def recall_of(win):
    tp = sum(a for a, _ in win)
    fn = sum(b for _, b in win)
    return tp / (tp + fn) if (tp + fn) else np.nan


# --------------------------------------------------------------- dashboard
def build_dashboard(headless):
    import matplotlib
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9.2))
    gs = GridSpec(3, 3, height_ratios=[0.4, 1, 1], width_ratios=[1, 1, 0.75],
                  hspace=0.42, wspace=0.22, left=0.05, right=0.985, top=0.97, bottom=0.07)

    a_ban = fig.add_subplot(gs[0, :])
    a_ban.axis("off")
    ban_box = a_ban.text(0.0, 0.95, "", transform=a_ban.transAxes, va="top", ha="left",
                         fontsize=13, family="monospace",
                         bbox=dict(boxstyle="round,pad=0.6", fc="#222", ec="#2ecc71", lw=2))

    a_blk = fig.add_subplot(gs[1, 0])
    a_rec = fig.add_subplot(gs[1, 1])
    a_psi = fig.add_subplot(gs[2, 0])
    a_cost = fig.add_subplot(gs[2, 1])
    a_log = fig.add_subplot(gs[1:, 2])

    C_S, C_O = "#ff6b6b", "#4ecdc4"
    for ax in (a_blk, a_rec, a_psi, a_cost):
        ax.set_xlim(0, N_STEPS)
        ax.set_xlabel("tempo (horas)", fontsize=9)
        for s, e, _, _, col in ACTS:
            ax.axvspan(s, e, color=col, alpha=0.10, lw=0)

    a_blk.set_title("Quanto estamos bloqueando?  (sem rótulo)", fontsize=11)
    a_blk.set_ylabel("% das transações bloqueadas")
    a_blk.set_ylim(0, 60)
    (b_s,) = a_blk.plot([], [], color=C_S, lw=2, label="Estático")
    (b_o,) = a_blk.plot([], [], color=C_O, lw=2, label="Online")
    a_blk.axhline(7, color="white", ls=":", lw=1)
    a_blk.text(1, 8.5, "normal ≈ 7%", fontsize=8, color="white")
    a_blk.legend(loc="upper left", fontsize=8)

    a_rec.set_title("Fraudes capturadas — recall  (rótulo chega com atraso)", fontsize=11)
    a_rec.set_ylabel("recall")
    a_rec.set_ylim(0, 1.05)
    a_rec.axhline(RECALL_SLO, color="#ffd93d", ls="--", lw=1.3)
    a_rec.text(1, RECALL_SLO + 0.02, "meta mínima (SLO) 70%", fontsize=8, color="#ffd93d")
    (r_s,) = a_rec.plot([], [], color=C_S, lw=2, label="Estático")
    (r_o,) = a_rec.plot([], [], color=C_O, lw=2, label="Online")
    a_rec.legend(loc="lower left", fontsize=8)

    a_psi.set_title("Mudou o perfil das transações?  PSI  (sem rótulo)", fontsize=11)
    a_psi.set_ylabel("PSI (>0.2 = mudou)")
    a_psi.set_ylim(0, 2.5)
    a_psi.axhline(PSI_LIMIT, color="#ffd93d", ls="--", lw=1.3)
    psi_lines = [a_psi.plot([], [], lw=1.8, label=f)[0] for f in FEATURES]
    a_psi.legend(loc="upper left", fontsize=7, ncol=2)

    a_cost.set_title("Prejuízo acumulado  (fraude que passou + clientes bons bloqueados)",
                     fontsize=10)
    a_cost.set_ylabel("R$ mil")
    c_s, = a_cost.plot([], [], color=C_S, lw=2.4, label="Estático")
    c_o, = a_cost.plot([], [], color=C_O, lw=2.4, label="Online")
    a_cost.legend(loc="upper left", fontsize=8)
    cost_txt = a_cost.text(0.98, 0.04, "", transform=a_cost.transAxes, ha="right",
                           fontsize=11, fontweight="bold")

    a_log.set_title("Alertas e ações do sistema", fontsize=11)
    a_log.axis("off")
    log_txt = a_log.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=8.6,
                         transform=a_log.transAxes, wrap=True)

    ctrl = {"restart": False}
    keep = []                                  # evita o botao ser coletado pelo GC
    if not headless:
        from matplotlib.widgets import Button
        ax_btn = fig.add_axes([0.865, 0.012, 0.12, 0.05])
        btn = Button(ax_btn, "↻ Reiniciar (R)", color="#333", hovercolor="#555")
        btn.label.set_color("white")
        btn.on_clicked(lambda _e: ctrl.update(restart=True))
        fig.canvas.mpl_connect("key_press_event",
                               lambda e: ctrl.update(restart=True) if e.key in ("r", "R") else None)
        keep.append(btn)
        plt.ion()
        plt.show(block=False)
    return plt, fig, dict(a_ban=a_ban, ban=ban_box, b_s=b_s, b_o=b_o, r_s=r_s, r_o=r_o,
                          psi=psi_lines, c_s=c_s, c_o=c_o, cost_txt=cost_txt,
                          log=log_txt, a_cost=a_cost, ctrl=ctrl, keep=keep)


def reset_dashboard(g):
    for ln in [g["b_s"], g["b_o"], g["r_s"], g["r_o"], g["c_s"], g["c_o"], *g["psi"]]:
        ln.set_data([], [])
    g["cost_txt"].set_text("")
    g["log"].set_text("sistema saudável")
    g["ban"].set_text("")
    g["a_cost"].set_ylim(0, 10)


def wrap(s, w=34):
    out = []
    for line in s.split("\n"):
        while len(line) > w:
            cut = line.rfind(" ", 0, w)
            cut = cut if cut > 0 else w
            out.append(line[:cut])
            line = "   " + line[cut:].lstrip()
        out.append(line)
    return "\n".join(out)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=0.15, help="segundos por hora simulada")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    plt, fig, g = build_dashboard(args.headless)
    while simulate(args, plt, fig, g) and not args.headless:
        log(f"\n{BLD}↻ Reiniciando simulação...{RST}\n", CYA)


def simulate(args, plt, fig, g):
    """Roda a simulacao. Retorna True se o usuario pediu para reiniciar."""
    ctrl = g["ctrl"]
    ctrl["restart"] = False
    reset_dashboard(g)
    rng = np.random.default_rng(SEED)
    bank = Bank(rng)

    # ---- treino inicial: so dados de um dia normal
    tr = [bank.batch(0) for _ in range(80)]
    X_tr = np.vstack([t[0] for t in tr])
    y_tr = np.concatenate([t[1] for t in tr])
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-9
    scale = lambda X: (X - mu) / sd

    def make():
        m = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=LR_NORMAL,
                          alpha=1e-4, class_weight={0: 1, 1: 4}, random_state=SEED)
        m.fit(scale(X_tr), y_tr)
        return m

    static, online = make(), make()
    base_block = float(online.predict(scale(X_tr)).mean())

    log(f"{BLD}Banco: modelo de fraude treinado com {len(y_tr)} transações normais.{RST}", GRN)
    log(f"Taxa de bloqueio normal ≈ {base_block*100:.1f}%\n", DIM)

    # ---- estado
    pending = deque()                     # transacoes esperando o chargeback
    win_X = deque(maxlen=WINDOW)
    rec_win = {"s": deque(maxlen=WINDOW), "o": deque(maxlen=WINDOW)}
    blk_win = deque(maxlen=WINDOW)
    H = {k: [] for k in ["t", "bs", "bo", "rs", "ro", "cs", "co"]}
    H["psi"] = [[] for _ in FEATURES]
    cost = {"s": 0.0, "o": 0.0}
    events = deque(maxlen=16)
    state = {"psi": False, "rec": False, "blk": False}
    boost_left = 0
    seen_act = None
    first = {}

    def event(step, text, color=""):
        events.append(f"h{step:03d} {text}")
        log(f"[hora {step:3d}] {text}", color)

    for step in range(N_STEPS):
        act = act_of(step)
        if act[2] != seen_act:
            seen_act = act[2]
            log(f"\n{BLD}━━━ {act[2]} ━━━{RST}", CYA)
            log(act[3].replace("\n", " "), DIM)
            g["ban"].set_text(f"{act[2]}\n{act[3]}")
            g["ban"].get_bbox_patch().set_edgecolor(act[4])

        X, y = bank.batch(step)
        Z = scale(X)
        ps, po = static.predict(Z), online.predict(Z)
        valor = np.exp(X[:, 0])

        # ---- prejuizo imediato (o simulador conhece a verdade; o banco nao)
        for tag, p in (("s", ps), ("o", po)):
            miss = (y == 1) & (p == 0)
            missed = valor[miss].sum() + miss.sum() * FRAUD_FIXED_COST
            blocked_good = ((y == 0) & (p == 1)).sum() * FP_COST
            cost[tag] += missed + blocked_good

        # ---- sinais SEM rotulo
        blk_s, blk_o = ps.mean() * 100, po.mean() * 100
        blk_win.append(po.mean())
        win_X.append(X)
        Xw = np.vstack(win_X)
        psis = [psi(X_tr[:, j], Xw[:, j]) for j in range(len(FEATURES))]

        # ---- rotulos chegam com atraso: avalia recall e ATUALIZA o modelo online
        pending.append((step + LABEL_DELAY, Z, y, ps, po))
        while pending and pending[0][0] <= step:
            _, Zl, yl, psl, pol = pending.popleft()
            pos = yl == 1
            rec_win["s"].append(((psl[pos] == 1).sum(), (psl[pos] == 0).sum()))
            rec_win["o"].append(((pol[pos] == 1).sum(), (pol[pos] == 0).sum()))
            online.partial_fit(Zl, yl)
        rec_s, rec_o = recall_of(rec_win["s"]), recall_of(rec_win["o"])

        # ---- ALERTAS (sobre o modelo em producao = online)
        worst = int(np.argmax(psis))
        psi_bad = max(psis) > PSI_LIMIT and step >= WINDOW
        blk_ratio = float(np.mean(blk_win)) / base_block
        blk_bad = step >= WINDOW and (blk_ratio > BLOCK_RATE_FACTOR or blk_ratio < 1 / BLOCK_RATE_FACTOR)
        rec_bad = (not np.isnan(rec_o)) and rec_o < RECALL_SLO

        new_alert = False
        if psi_bad and not state["psi"]:
            event(step, f"ALERTA: perfil dos dados mudou ({FEATURES[worst]}, PSI={max(psis):.2f})", YEL)
            first.setdefault("psi", step); new_alert = True
        if not psi_bad and state["psi"]:
            event(step, "OK: perfil dos dados voltou ao normal", GRN)
        if blk_bad and not state["blk"]:
            event(step, f"ALERTA: taxa de bloqueio anormal ({np.mean(blk_win)*100:.0f}% vs {base_block*100:.0f}%)", YEL)
            first.setdefault("blk", step); new_alert = True
        if not blk_bad and state["blk"]:
            event(step, "OK: taxa de bloqueio normalizada", GRN)
        if rec_bad and not state["rec"]:
            event(step, f"CRITICO: recall {rec_o*100:.0f}% abaixo da meta de {RECALL_SLO*100:.0f}%", RED)
            first.setdefault("rec", step); new_alert = True
        if (not rec_bad) and state["rec"] and not np.isnan(rec_o) and rec_o > RECALL_SLO + 0.1:
            event(step, f"OK: recall recuperado ({rec_o*100:.0f}%)", GRN)
            state["rec"] = False
        if rec_bad:
            state["rec"] = True
        state["psi"], state["blk"] = psi_bad, blk_bad

        if new_alert and boost_left == 0:
            boost_left = BOOST_STEPS
            online.set_params(eta0=LR_BOOST)
            event(step, f"ACAO: modelo online aprende mais rapido (lr {LR_NORMAL}->{LR_BOOST})", CYA)
        elif boost_left > 0:
            boost_left -= 1
            if boost_left == 0:
                online.set_params(eta0=LR_NORMAL)
                event(step, "ACAO: aprendizado volta ao ritmo normal", DIM)

        # ---- historico e grafico
        H["t"].append(step)
        H["bs"].append(blk_s); H["bo"].append(blk_o)
        H["rs"].append(rec_s); H["ro"].append(rec_o)
        H["cs"].append(cost["s"] / 1000); H["co"].append(cost["o"] / 1000)
        for j in range(len(FEATURES)):
            H["psi"][j].append(psis[j])
        g["b_s"].set_data(H["t"], H["bs"]); g["b_o"].set_data(H["t"], H["bo"])
        g["r_s"].set_data(H["t"], H["rs"]); g["r_o"].set_data(H["t"], H["ro"])
        g["c_s"].set_data(H["t"], H["cs"]); g["c_o"].set_data(H["t"], H["co"])
        g["a_cost"].set_ylim(0, max(10, max(H["cs"]) * 1.1))
        for j, ln in enumerate(g["psi"]):
            ln.set_data(H["t"], H["psi"][j])
        g["cost_txt"].set_text(f"Estático: R$ {cost['s']/1000:,.0f} mil\nOnline:   R$ {cost['o']/1000:,.0f} mil")
        g["log"].set_text("\n".join(wrap(e) for e in events) if events else "sistema saudável")
        if not args.headless:
            if not plt.fignum_exists(fig.number):
                return False
            plt.pause(args.speed)
            if ctrl["restart"]:
                return True

    # ---- resumo
    saved = cost["s"] - cost["o"]
    log(f"\n{BLD}━━━ RESUMO ━━━{RST}", CYA)
    log(f"Prejuízo acumulado: estático R$ {cost['s']:,.0f}  |  online R$ {cost['o']:,.0f}", GRN)
    log(f"Economia por atualizar o modelo online: R$ {saved:,.0f} ({saved/cost['s']*100:.0f}%)", GRN)
    for k, nome in (("psi", "mudança de perfil (PSI)"), ("blk", "taxa de bloqueio"), ("rec", "queda do recall")):
        if k in first:
            log(f"Primeiro alerta de {nome}: hora {first[k]}")
    log("Lição: observabilidade avisa que algo mudou; atualização online corrige.\n", CYA)

    if args.headless:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_fraude.png")
        fig.savefig(out, dpi=120)
        log(f"Dashboard salvo em {out}")
        return False
    log("Pressione R (ou clique em Reiniciar) para rodar de novo; feche a janela para sair.", DIM)
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)
        if ctrl["restart"]:
            return True
    return False


if __name__ == "__main__":
    main()
